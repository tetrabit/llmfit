# Multi-stage build for llmfit

# Stage 1: Build web dashboard assets
FROM node:22-slim AS web-builder

WORKDIR /web

COPY llmfit-web/package.json llmfit-web/package-lock.json ./
RUN npm ci

COPY llmfit-web/ ./
RUN npm run build

# Stage 2: Build the Rust binary
FROM rust:1.88-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy workspace configuration
COPY Cargo.toml Cargo.lock ./

# Copy workspace members needed for the build
COPY llmfit-core/ ./llmfit-core/
COPY llmfit-tui/ ./llmfit-tui/
COPY data/ ./data/

# Stub out llmfit-desktop so Cargo can resolve the workspace
# (it's a Tauri app, not built in Docker)
COPY llmfit-desktop/Cargo.toml ./llmfit-desktop/Cargo.toml
RUN mkdir -p llmfit-desktop/src && echo "fn main() {}" > llmfit-desktop/src/main.rs

# Copy pre-built web assets from the web-builder stage
COPY --from=web-builder /web/dist/ ./llmfit-web/dist/

# Build release binary for llmfit-tui
RUN cargo build --release -p llmfit

# Stage 3: Runtime image
FROM debian:bookworm-slim

# Install runtime dependencies for hardware detection
RUN apt-get update && apt-get install -y \
    pciutils \
    lshw \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary from builder
COPY --from=builder /build/target/release/llmfit /usr/local/bin/llmfit

# Create a non-root user
RUN useradd -m -u 1000 llmfit && \
    chown -R llmfit:llmfit /usr/local/bin/llmfit

USER llmfit

# Set default command to output JSON recommendations
# In Kubernetes, this will run once per node and log results
ENTRYPOINT ["/usr/local/bin/llmfit"]
CMD ["recommend", "--json"]
