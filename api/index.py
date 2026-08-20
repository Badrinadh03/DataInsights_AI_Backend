from functions.main import app, health as _health


class VercelPathMiddleware:
	"""Allow Flask routes to work with or without Vercel's /api prefix."""

	def __init__(self, application):
		self.application = application

	def __call__(self, environ, start_response):
		path = environ.get("PATH_INFO", "")
		if path == "/api" or path.startswith("/api/"):
			environ["PATH_INFO"] = path[4:] or "/"
		return self.application(environ, start_response)


app.wsgi_app = VercelPathMiddleware(app.wsgi_app)

# Vercel Python serverless entrypoint
# The Flask app is defined in functions/main.py and is re-exported here
# so Vercel can discover the app correctly.
# Vercel mounts this function at /api; the middleware above maps /api/* to the
# routes defined by the application. The function root is also a health check.
app.add_url_rule("/", view_func=_health, methods=["GET"])
