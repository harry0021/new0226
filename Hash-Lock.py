import hashlib
import random

class Tag:
    def __init__(self, id):
        self.id = id
        self.secret_key = hashlib.sha256(str(random.random()).encode()).hexdigest()
        self.challenge = None
        self.response = None

    def send_challenge(self):
        self.challenge = hashlib.sha256(str(random.random()).encode()).hexdigest()
        return self.challenge

    def receive_response(self, response):
        self.response = response

class Reader:
    def __init__(self, id):
        self.id = id

    def query(self, tag):
        challenge = tag.send_challenge()
        response = self.send(challenge)
        tag.receive_response(response)

    def send(self, challenge):
        return challenge

class Database:
    def __init__(self):
        self.data = {}

    def search(self, tag):
        if tag.id in self.data:
            if hashlib.sha256(self.data[tag.id].encode()).hexdigest() == tag.response:
                return self.data[tag.id]
            else:
                return "Access denied"
        else:
            return "Tag not found"

    def add_tag_data(self, tag_id, data):
        self.data[tag_id] = data

# Example usage:
database = Database()
database.add_tag_data("tag1", "data1")
database.add_tag_data("tag2", "data2")

tag1 = Tag("tag1")
reader1 = Reader("reader1")
reader1.query(tag1)
print(database.search(tag1))

tag2 = Tag("tag2")
reader2 = Reader("reader2")
reader2.query(tag2)
print(database.search(tag2))
