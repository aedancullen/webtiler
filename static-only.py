from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import socket

class Handler(BaseHTTPRequestHandler):
	
	def handle(self):
		try:
			return BaseHTTPRequestHandler.handle(self)
		except (socket.error, socket.timeout) as e:
			pass
		
	def do_GET(self):
		self.send_response(200)
		self.send_header("Content-type", "text/html")
		self.end_headers()
		with open("index.html", "rb") as fh:
			self.wfile.write(fh.read())

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
	pass

if __name__ == '__main__':
	server = ThreadedHTTPServer(("", 8080), Handler)
	server.serve_forever()
