# Hosting on Nebius AI Cloud

The agent talks to [Nebius Token Factory](https://tokenfactory.nebius.com/) for
candidates and for grid-from-photo. This box only serves the FastAPI + Vite
page. No GPU. Do not put `NEBIUS_API_KEY` in the image.

## Machine

1. In the [Nebius console](https://console.nebius.com), create a **standalone**
   Compute VM in the same project as Token Factory (typically `eu-north1`).
2. Platform `cpu-e2`, preset `2vcpu-8gb`, Ubuntu, ~20 GiB boot disk.
3. Public IPv4, security group open on **22**, **80**, and **443**.
4. SSH in (not as `root` or `admin`).

Skip Managed Kubernetes and Container VMs. One in-memory solve job does not
need a cluster, and Container VMs make TLS + a secret file awkward.

## Run

```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 git
sudo usermod -aG docker "$USER"   # then log out and back in
git clone <this repo> && cd crossword-solver
printf 'NEBIUS_API_KEY=%s\n' 'paste-the-token-factory-key' > crossword.env
chmod 600 crossword.env
docker compose -f deploy/docker-compose.yml up -d --build
```

Open `http://<public-ip>/`. **Your puzzle** is the BYO tab. Mini still works
offline-style only if you also set oracle locally; on this host the key is
present, so Nebius is the default.

Health: `curl -s http://<public-ip>/api/health`

## HTTPS

Point a DNS A record at the static public IP. Replace `deploy/Caddyfile` with:

```
{$CROSSWORD_HOST} {
    reverse_proxy app:8000 {
        flush_interval -1
    }
}
```

Set `CROSSWORD_HOST=solver.example.com` in `crossword.env` (Caddy reads the
file via compose `env_file` only for `app`; export it for Caddy or bake the
hostname into the Caddyfile). Recreate: `docker compose -f deploy/docker-compose.yml up -d`.

A 15×15 live solve can run for minutes. `flush_interval -1` keeps SSE from
buffering. The process is one solve at a time, 5 ingest-or-solve starts per
client IP per hour, and `DAILY_SOLVE_CAP` (default 40) for the whole process.

## Updates

```bash
git pull
docker compose -f deploy/docker-compose.yml up -d --build
```
