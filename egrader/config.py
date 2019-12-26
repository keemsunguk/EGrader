import json


class Config:
    """
    Project Configuration IO
    """

    def __init__(self):
        try:
            with open("/Users/keemsunguk/Projects/EssayGrader/config/config.json", "r") as confin:
                self.conf = json.load(confin)
        except Exception as e:
            print("Config Error: Check {ProjectRoot}/config/config.json: %s", str(e))
            # default
            self.conf = {
                "Projects": "/Users/keemsunguk/Projects/",
                "EGraderRoot": "EssayGrader/",
                "RemoteMongo": "mongodb+srv://{uid}:{password}@cluster0-ccgud.gcp.mongodb.net",
                "LocalMongo": "mongodb://localhost:27017/",
            }

    def get_config(self):
        """
        Get configuration
        :return: conf dictionary
        """
        return self.conf

    def __str__(self):
        return str(self.conf)


if __name__ != '__main__':
    config = Config()
