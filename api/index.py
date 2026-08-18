from functions.main import app, health as _health

# Vercel Python serverless entrypoint
# The Flask app is defined in functions/main.py and is re-exported here
# so Vercel can discover the app correctly.
# Add compatibility aliases because Vercel mounts the function at /api.
app.add_url_rule("/health", view_func=_health, methods=["GET"])
app.add_url_rule("/api/health", view_func=_health, methods=["GET"])
app.add_url_rule("/api", view_func=_health, methods=["GET"])
