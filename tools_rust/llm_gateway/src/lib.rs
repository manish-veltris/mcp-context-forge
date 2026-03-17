use std::collections::HashMap;
use std::env;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_stream::stream;
use axum::body::Body;
use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tracing::{error, warn};
use uuid::Uuid;

const INTERNAL_FORWARD_HEADER: &str = "x-forwarded-internally";
const INTERNAL_SECRET_HEADER: &str = "x-contextforge-internal-secret";

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub bind_addr: String,
    pub core_url: String,
    pub internal_secret: Option<String>,
    pub request_timeout_secs: u64,
    pub core_insecure_tls: bool,
}

impl AppConfig {
    pub fn from_env() -> Self {
        Self {
            bind_addr: env::var("LLM_GATEWAY_BIND")
                .unwrap_or_else(|_| "127.0.0.1:8011".to_string()),
            core_url: env::var("LLM_GATEWAY_CORE_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:4444".to_string()),
            internal_secret: env::var("LLM_GATEWAY_INTERNAL_SECRET")
                .ok()
                .filter(|value| !value.is_empty()),
            request_timeout_secs: env::var("LLM_GATEWAY_REQUEST_TIMEOUT_SECONDS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(120),
            core_insecure_tls: env::var("LLM_GATEWAY_CORE_INSECURE_TLS")
                .ok()
                .map(|value| {
                    matches!(
                        value.to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(false),
        }
    }
}

#[derive(Clone)]
pub struct AppState {
    upstream_client: Client,
    core_client: Client,
    config: AppConfig,
}

impl AppState {
    pub fn new(config: AppConfig) -> Result<Self, String> {
        let upstream_client = Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .build()
            .map_err(|error| format!("failed to build upstream reqwest client: {error}"))?;
        let core_client = Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .danger_accept_invalid_certs(config.core_insecure_tls)
            .build()
            .map_err(|error| format!("failed to build core reqwest client: {error}"))?;
        Ok(Self {
            upstream_client,
            core_client,
            config,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(default = "default_tool_type")]
    r#type: String,
    function: FunctionDefinition,
}

fn default_tool_type() -> String {
    "function".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    role: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<Value>>,
    #[serde(default)]
    tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    max_tokens: Option<u64>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    tools: Option<Vec<ToolDefinition>>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    frequency_penalty: Option<f64>,
    #[serde(default)]
    presence_penalty: Option<f64>,
    #[serde(default)]
    stop: Option<Value>,
    #[serde(default)]
    user: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResolveChatCompletionTargetRequest {
    model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedChatCompletionTarget {
    runtime_kind: String,
    upstream_url: String,
    #[serde(default)]
    upstream_headers: HashMap<String, String>,
    model_id: String,
    #[serde(default)]
    default_temperature: Option<f64>,
    #[serde(default)]
    default_max_tokens: Option<u64>,
}

pub fn build_app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ready", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

async fn list_models(State(state): State<AppState>, headers: HeaderMap) -> Response<Body> {
    if let Err(response) = ensure_trusted_request(&headers, state.config.internal_secret.as_deref())
    {
        return response;
    }

    match call_core_models(&state).await {
        Ok(response) => response,
        Err(response) => response,
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response<Body> {
    if let Err(response) = ensure_trusted_request(&headers, state.config.internal_secret.as_deref())
    {
        return response;
    }

    let target = match resolve_chat_target(&state, &request.model).await {
        Ok(target) => target,
        Err(response) => return response,
    };

    if request.stream {
        handle_streaming_request(state, request, target).await
    } else {
        handle_non_streaming_request(&state, &request, &target).await
    }
}

fn ensure_trusted_request(
    headers: &HeaderMap,
    expected_secret: Option<&str>,
) -> Result<(), Response<Body>> {
    let forwarded = headers
        .get(INTERNAL_FORWARD_HEADER)
        .and_then(|value| value.to_str().ok())
        == Some("true");
    if !forwarded {
        return Err(error_response(
            StatusCode::FORBIDDEN,
            "Trusted runtime access required",
        ));
    }

    if let Some(expected_secret) = expected_secret {
        let provided = headers
            .get(INTERNAL_SECRET_HEADER)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("");
        if provided != expected_secret {
            return Err(error_response(
                StatusCode::FORBIDDEN,
                "Trusted runtime access required",
            ));
        }
    }

    Ok(())
}

async fn call_core_models(state: &AppState) -> Result<Response<Body>, Response<Body>> {
    let response = state
        .core_client
        .get(format!(
            "{}/_internal/rust/llm/models",
            state.config.core_url.trim_end_matches('/')
        ))
        .headers(build_internal_headers(
            state.config.internal_secret.as_deref(),
        ))
        .send()
        .await
        .map_err(|error| {
            error!("Rust LLM Gateway failed to fetch models from core: {error}");
            error_response(StatusCode::BAD_GATEWAY, "Core LLM catalog unavailable")
        })?;

    let status = response.status();
    let body = response.bytes().await.map_err(|error| {
        error!("Rust LLM Gateway failed to read core model response: {error}");
        error_response(StatusCode::BAD_GATEWAY, "Core LLM catalog unavailable")
    })?;

    Ok(response_with_bytes(status, "application/json", body))
}

async fn resolve_chat_target(
    state: &AppState,
    model: &str,
) -> Result<ResolvedChatCompletionTarget, Response<Body>> {
    let response = state
        .core_client
        .post(format!(
            "{}/_internal/rust/llm/resolve-chat-target",
            state.config.core_url.trim_end_matches('/')
        ))
        .headers(build_internal_headers(
            state.config.internal_secret.as_deref(),
        ))
        .json(&ResolveChatCompletionTargetRequest {
            model: model.to_string(),
        })
        .send()
        .await
        .map_err(|error| {
            error!("Rust LLM Gateway failed to resolve target in core: {error}");
            error_response(
                StatusCode::BAD_GATEWAY,
                "Core LLM target resolution unavailable",
            )
        })?;

    if !response.status().is_success() {
        return Err(proxy_json_error_response(response).await);
    }

    response
        .json::<ResolvedChatCompletionTarget>()
        .await
        .map_err(|error| {
            error!("Rust LLM Gateway failed to decode core target response: {error}");
            error_response(
                StatusCode::BAD_GATEWAY,
                "Core LLM target resolution unavailable",
            )
        })
}

async fn handle_non_streaming_request(
    state: &AppState,
    request: &ChatCompletionRequest,
    target: &ResolvedChatCompletionTarget,
) -> Response<Body> {
    let body = build_request_body(request, target);
    let response = match build_upstream_request(state, target)
        .json(&body)
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => {
            error!("Rust LLM Gateway upstream request failed: {error}");
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("Connection error: {error}"),
            );
        }
    };

    if !response.status().is_success() {
        return upstream_error_response(response).await;
    }

    let data = match response.json::<Value>().await {
        Ok(data) => data,
        Err(error) => {
            error!("Rust LLM Gateway failed to decode upstream JSON: {error}");
            return error_response(StatusCode::BAD_GATEWAY, "Invalid upstream JSON response");
        }
    };

    let transformed = match target.runtime_kind.as_str() {
        "anthropic" => transform_anthropic_response(&data, &target.model_id),
        "ollama_native" => transform_ollama_response(&data, &target.model_id),
        _ => data,
    };

    Json(transformed).into_response()
}

async fn handle_streaming_request(
    state: AppState,
    request: ChatCompletionRequest,
    target: ResolvedChatCompletionTarget,
) -> Response<Body> {
    let body = build_request_body(&request, &target);
    let response = match build_upstream_request(&state, &target)
        .json(&body)
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => {
            error!("Rust LLM Gateway upstream stream failed: {error}");
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("Connection error: {error}"),
            );
        }
    };

    if !response.status().is_success() {
        return upstream_error_response(response).await;
    }

    if matches!(
        target.runtime_kind.as_str(),
        "openai" | "azure_openai" | "ollama_openai"
    ) {
        let stream = response.bytes_stream().map(|result| {
            result
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error.to_string()))
        });
        return response_with_stream(stream);
    }

    let response_id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let created = unix_timestamp();
    let model_id = target.model_id.clone();
    let runtime_kind = target.runtime_kind.clone();
    let mut upstream = response.bytes_stream();

    let output = stream! {
        let mut buffer: Vec<u8> = Vec::new();

        while let Some(chunk) = upstream.next().await {
            match chunk {
                Ok(bytes) => {
                    buffer.extend_from_slice(&bytes);
                    for line in collect_transformed_lines(&mut buffer, &runtime_kind, &response_id, created, &model_id, false) {
                        yield Ok::<Bytes, std::io::Error>(Bytes::from(line));
                    }
                }
                Err(error) => {
                    let io_error = std::io::Error::new(std::io::ErrorKind::Other, error.to_string());
                    yield Err(io_error);
                    return;
                }
            }
        }

        if !buffer.is_empty() {
            for line in collect_transformed_lines(&mut buffer, &runtime_kind, &response_id, created, &model_id, true) {
                yield Ok::<Bytes, std::io::Error>(Bytes::from(line));
            }
        }
    };

    response_with_stream(output)
}

fn build_internal_headers(secret: Option<&str>) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static(INTERNAL_FORWARD_HEADER),
        HeaderValue::from_static("true"),
    );
    if let Some(secret) = secret {
        if let Ok(value) = HeaderValue::from_str(secret) {
            headers.insert(HeaderName::from_static(INTERNAL_SECRET_HEADER), value);
        }
    }
    headers
}

fn build_upstream_request<'a>(
    state: &'a AppState,
    target: &'a ResolvedChatCompletionTarget,
) -> reqwest::RequestBuilder {
    let mut request = state.upstream_client.post(&target.upstream_url);
    let mut headers = HeaderMap::new();
    for (name, value) in &target.upstream_headers {
        if let (Ok(header_name), Ok(header_value)) = (
            HeaderName::from_bytes(name.as_bytes()),
            HeaderValue::from_str(value),
        ) {
            headers.insert(header_name, header_value);
        }
    }
    request = request.headers(headers);
    request
}

fn build_request_body(
    request: &ChatCompletionRequest,
    target: &ResolvedChatCompletionTarget,
) -> Value {
    match target.runtime_kind.as_str() {
        "azure_openai" => build_azure_body(request, target),
        "anthropic" => build_anthropic_body(request, target),
        "ollama_native" => build_ollama_native_body(request, target),
        _ => build_openai_body(request, target),
    }
}

fn build_openai_body(
    request: &ChatCompletionRequest,
    target: &ResolvedChatCompletionTarget,
) -> Value {
    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(target.model_id.clone()));
    body.insert(
        "messages".to_string(),
        Value::Array(
            request
                .messages
                .iter()
                .map(serialize_chat_message)
                .collect::<Vec<Value>>(),
        ),
    );
    insert_optional_number(
        &mut body,
        "temperature",
        request.temperature.or(target.default_temperature),
    );
    insert_optional_u64(
        &mut body,
        "max_tokens",
        request.max_tokens.or(target.default_max_tokens),
    );
    body.insert("stream".to_string(), Value::Bool(request.stream));
    if let Some(tools) = &request.tools {
        body.insert(
            "tools".to_string(),
            Value::Array(
                tools
                    .iter()
                    .map(|tool| serde_json::to_value(tool).unwrap_or(Value::Null))
                    .collect(),
            ),
        );
    }
    if let Some(tool_choice) = &request.tool_choice {
        body.insert("tool_choice".to_string(), tool_choice.clone());
    }
    insert_optional_number(&mut body, "top_p", request.top_p);
    insert_optional_number(&mut body, "frequency_penalty", request.frequency_penalty);
    insert_optional_number(&mut body, "presence_penalty", request.presence_penalty);
    if let Some(stop) = &request.stop {
        body.insert("stop".to_string(), stop.clone());
    }
    if let Some(user) = &request.user {
        body.insert("user".to_string(), Value::String(user.clone()));
    }
    Value::Object(body)
}

fn build_azure_body(
    request: &ChatCompletionRequest,
    target: &ResolvedChatCompletionTarget,
) -> Value {
    let mut body = Map::new();
    body.insert(
        "messages".to_string(),
        Value::Array(
            request
                .messages
                .iter()
                .map(serialize_chat_message)
                .collect::<Vec<Value>>(),
        ),
    );
    insert_optional_number(
        &mut body,
        "temperature",
        request.temperature.or(target.default_temperature),
    );
    insert_optional_u64(
        &mut body,
        "max_tokens",
        request.max_tokens.or(target.default_max_tokens),
    );
    body.insert("stream".to_string(), Value::Bool(request.stream));
    Value::Object(body)
}

fn build_anthropic_body(
    request: &ChatCompletionRequest,
    target: &ResolvedChatCompletionTarget,
) -> Value {
    let mut body = Map::new();
    let mut system_message: Option<String> = None;
    let mut messages: Vec<Value> = Vec::new();

    for message in &request.messages {
        if message.role == "system" {
            system_message = message.content.clone();
            continue;
        }
        messages.push(json!({
            "role": message.role,
            "content": message.content.clone().unwrap_or_default(),
        }));
    }

    body.insert("model".to_string(), Value::String(target.model_id.clone()));
    body.insert("messages".to_string(), Value::Array(messages));
    body.insert(
        "max_tokens".to_string(),
        Value::Number(serde_json::Number::from(
            request
                .max_tokens
                .or(target.default_max_tokens)
                .unwrap_or(4096),
        )),
    );
    if let Some(system) = system_message {
        body.insert("system".to_string(), Value::String(system));
    }
    insert_optional_number(
        &mut body,
        "temperature",
        request.temperature.or(target.default_temperature),
    );
    body.insert("stream".to_string(), Value::Bool(request.stream));
    Value::Object(body)
}

fn build_ollama_native_body(
    request: &ChatCompletionRequest,
    target: &ResolvedChatCompletionTarget,
) -> Value {
    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(target.model_id.clone()));
    body.insert(
        "messages".to_string(),
        Value::Array(
            request
                .messages
                .iter()
                .map(|message| {
                    json!({
                        "role": message.role,
                        "content": message.content.clone().unwrap_or_default(),
                    })
                })
                .collect(),
        ),
    );
    body.insert("stream".to_string(), Value::Bool(request.stream));

    let temperature = request.temperature.or(target.default_temperature);
    if let Some(temperature) = temperature {
        body.insert(
            "options".to_string(),
            json!({
                "temperature": temperature,
            }),
        );
    }
    Value::Object(body)
}

fn serialize_chat_message(message: &ChatMessage) -> Value {
    let mut object = Map::new();
    object.insert("role".to_string(), Value::String(message.role.clone()));
    if let Some(content) = &message.content {
        object.insert("content".to_string(), Value::String(content.clone()));
    }
    if let Some(name) = &message.name {
        object.insert("name".to_string(), Value::String(name.clone()));
    }
    if let Some(tool_calls) = &message.tool_calls {
        object.insert("tool_calls".to_string(), Value::Array(tool_calls.clone()));
    }
    if let Some(tool_call_id) = &message.tool_call_id {
        object.insert(
            "tool_call_id".to_string(),
            Value::String(tool_call_id.clone()),
        );
    }
    Value::Object(object)
}

fn insert_optional_number(map: &mut Map<String, Value>, key: &str, value: Option<f64>) {
    if let Some(value) = value {
        if let Some(number) = serde_json::Number::from_f64(value) {
            map.insert(key.to_string(), Value::Number(number));
        }
    }
}

fn insert_optional_u64(map: &mut Map<String, Value>, key: &str, value: Option<u64>) {
    if let Some(value) = value {
        map.insert(
            key.to_string(),
            Value::Number(serde_json::Number::from(value)),
        );
    }
}

fn transform_anthropic_response(data: &Value, model_id: &str) -> Value {
    let content = data
        .get("content")
        .and_then(Value::as_array)
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|block| {
                    if block.get("type").and_then(Value::as_str) == Some("text") {
                        block.get("text").and_then(Value::as_str)
                    } else {
                        None
                    }
                })
                .collect::<Vec<&str>>()
                .join("")
        })
        .unwrap_or_default();

    let input_tokens = data
        .get("usage")
        .and_then(|usage| usage.get("input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = data
        .get("usage")
        .and_then(|usage| usage.get("output_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);

    json!({
        "id": data.get("id").cloned().unwrap_or_else(|| Value::String(generate_chat_completion_id())),
        "object": "chat.completion",
        "created": unix_timestamp(),
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": data.get("stop_reason").cloned().unwrap_or_else(|| Value::String("stop".to_string())),
        }],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    })
}

fn transform_ollama_response(data: &Value, model_id: &str) -> Value {
    let role = data
        .get("message")
        .and_then(|message| message.get("role"))
        .and_then(Value::as_str)
        .unwrap_or("assistant");
    let content = data
        .get("message")
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let done = data.get("done").and_then(Value::as_bool).unwrap_or(false);
    let prompt_tokens = data
        .get("prompt_eval_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let completion_tokens = data.get("eval_count").and_then(Value::as_u64).unwrap_or(0);

    json!({
        "id": generate_chat_completion_id(),
        "object": "chat.completion",
        "created": unix_timestamp(),
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {
                "role": role,
                "content": content,
            },
            "finish_reason": if done { Value::String("stop".to_string()) } else { Value::Null },
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })
}

fn collect_transformed_lines(
    buffer: &mut Vec<u8>,
    runtime_kind: &str,
    response_id: &str,
    created: u64,
    model_id: &str,
    flush_remainder: bool,
) -> Vec<String> {
    let mut output = Vec::new();

    while let Some(position) = buffer.iter().position(|byte| *byte == b'\n') {
        let line_bytes = buffer.drain(..=position).collect::<Vec<u8>>();
        let line = String::from_utf8_lossy(&line_bytes)
            .trim_end_matches(&['\r', '\n'][..])
            .to_string();
        if line.is_empty() {
            continue;
        }
        let maybe_line = match runtime_kind {
            "anthropic" => transform_anthropic_stream_line(&line, response_id, created, model_id),
            "ollama_native" => transform_ollama_stream_line(&line, response_id, created, model_id),
            _ => None,
        };
        if let Some(line) = maybe_line {
            output.push(line);
        }
    }

    if flush_remainder && !buffer.is_empty() && !buffer.contains(&b'\n') {
        let line = String::from_utf8_lossy(buffer)
            .trim_end_matches(&['\r', '\n'][..])
            .to_string();
        buffer.clear();
        if line.is_empty() {
            return output;
        }
        let maybe_line = match runtime_kind {
            "anthropic" => transform_anthropic_stream_line(&line, response_id, created, model_id),
            "ollama_native" => transform_ollama_stream_line(&line, response_id, created, model_id),
            _ => None,
        };
        if let Some(line) = maybe_line {
            output.push(line);
        }
    }

    output
}

fn transform_anthropic_stream_line(
    line: &str,
    response_id: &str,
    created: u64,
    model_id: &str,
) -> Option<String> {
    if !line.starts_with("data:") {
        return None;
    }
    let data = line.trim_start_matches("data:").trim();
    if data == "[DONE]" {
        return Some("data: [DONE]\n\n".to_string());
    }
    let payload: Value = serde_json::from_str(data).ok()?;
    let event_type = payload.get("type").and_then(Value::as_str)?;

    match event_type {
        "content_block_delta" => {
            let delta_type = payload
                .get("delta")
                .and_then(|delta| delta.get("type"))
                .and_then(Value::as_str)?;
            if delta_type != "text_delta" {
                return None;
            }
            let text = payload
                .get("delta")
                .and_then(|delta| delta.get("text"))
                .and_then(Value::as_str)
                .unwrap_or("");
            Some(format!(
                "data: {}\n\n",
                json!({
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": Value::Null,
                    }],
                })
            ))
        }
        "message_stop" => Some(format!(
            "data: {}\n\n",
            json!({
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            })
        )),
        _ => None,
    }
}

fn transform_ollama_stream_line(
    line: &str,
    response_id: &str,
    created: u64,
    model_id: &str,
) -> Option<String> {
    let payload: Value = serde_json::from_str(line).ok()?;
    let done = payload
        .get("done")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let content = payload
        .get("message")
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .unwrap_or("");

    let chunk = if done {
        json!({
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        })
    } else {
        json!({
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": if content.is_empty() { json!({}) } else { json!({"content": content}) },
                "finish_reason": Value::Null,
            }],
        })
    };

    Some(format!("data: {}\n\n", chunk))
}

async fn upstream_error_response(response: reqwest::Response) -> Response<Body> {
    warn!(
        "Rust LLM Gateway upstream returned non-success status: {}",
        response.status()
    );
    error_response(
        StatusCode::BAD_GATEWAY,
        format!("Request failed: {}", response.status().as_u16()),
    )
}

async fn proxy_json_error_response(response: reqwest::Response) -> Response<Body> {
    let status = response.status();
    let detail = response
        .json::<Value>()
        .await
        .ok()
        .and_then(|value| {
            value
                .get("detail")
                .and_then(Value::as_str)
                .map(|value| value.to_string())
        })
        .unwrap_or_else(|| format!("Core LLM request failed: {}", status.as_u16()));
    error_response(status, detail)
}

fn response_with_stream<S>(stream: S) -> Response<Body>
where
    S: futures::Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static,
{
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream")
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to build stream response",
            )
        })
}

fn response_with_bytes(status: StatusCode, content_type: &str, body: Bytes) -> Response<Body> {
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, content_type)
        .body(Body::from(body))
        .unwrap_or_else(|_| {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to build response",
            )
        })
}

fn error_response(status: StatusCode, detail: impl Into<String>) -> Response<Body> {
    let payload = json!({
        "detail": detail.into(),
    });
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(payload.to_string()))
        .unwrap_or_else(|_| Response::new(Body::from("{\"detail\":\"internal error\"}")))
}

fn generate_chat_completion_id() -> String {
    format!("chatcmpl-{}", Uuid::new_v4().simple())
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(kind: &str) -> ResolvedChatCompletionTarget {
        ResolvedChatCompletionTarget {
            runtime_kind: kind.to_string(),
            upstream_url: "https://example.com/chat/completions".to_string(),
            upstream_headers: HashMap::new(),
            model_id: "gpt-4".to_string(),
            default_temperature: Some(0.7),
            default_max_tokens: Some(128),
        }
    }

    #[test]
    fn build_openai_body_uses_defaults() {
        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some("hi".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: None,
            max_tokens: None,
            stream: false,
            tools: None,
            tool_choice: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            user: None,
        };

        let body = build_openai_body(&request, &target("openai"));

        assert_eq!(body["temperature"], json!(0.7));
        assert_eq!(body["max_tokens"], json!(128));
        assert_eq!(body["model"], json!("gpt-4"));
    }

    #[test]
    fn build_anthropic_body_extracts_system_message() {
        let request = ChatCompletionRequest {
            model: "claude".to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: Some("sys".to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: Some("hi".to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
            temperature: None,
            max_tokens: None,
            stream: true,
            tools: None,
            tool_choice: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            user: None,
        };

        let body = build_anthropic_body(&request, &target("anthropic"));

        assert_eq!(body["system"], json!("sys"));
        assert_eq!(body["messages"][0]["role"], json!("user"));
        assert_eq!(body["stream"], json!(true));
    }

    #[test]
    fn transform_anthropic_response_merges_text() {
        let data = json!({
            "id": "resp",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "text", "text": " there"}
            ],
            "usage": {"input_tokens": 1, "output_tokens": 2},
            "stop_reason": "stop",
        });

        let transformed = transform_anthropic_response(&data, "claude");

        assert_eq!(
            transformed["choices"][0]["message"]["content"],
            json!("hi there")
        );
        assert_eq!(transformed["usage"]["total_tokens"], json!(3));
    }

    #[test]
    fn transform_ollama_stream_line_formats_sse() {
        let line = r#"{"message":{"content":"hi"},"done":false}"#;
        let transformed = transform_ollama_stream_line(line, "id", 1, "llama3").unwrap();

        assert!(transformed.starts_with("data: "));
        assert!(transformed.contains("\"content\":\"hi\""));
    }
}
