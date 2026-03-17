use tokio::net::TcpListener;
use tracing::info;
use tracing_subscriber::EnvFilter;

use llm_gateway::{AppConfig, AppState, build_app};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = AppConfig::from_env();
    let bind_addr = config.bind_addr.clone();
    let state = match AppState::new(config) {
        Ok(state) => state,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };

    let listener = match TcpListener::bind(&bind_addr).await {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("failed to bind {bind_addr}: {error}");
            std::process::exit(1);
        }
    };

    info!("Rust LLM Gateway listening on {}", bind_addr);
    if let Err(error) = axum::serve(listener, build_app(state)).await {
        eprintln!("Rust LLM Gateway server error: {error}");
        std::process::exit(1);
    }
}
