import json
import os

try:
    PROJECT_ROOT = os.environ['PROJECT_ROOT']
except:
    PROJECT_ROOT = '/Users/keemsunguk/Projects/EssayGrader/'

class Config:
    """
    Project Configuration IO
    """

    def __init__(self):
        try:
            with open(PROJECT_ROOT+"/config/config.json", "r") as confin:
                self.conf = json.load(confin)
        except Exception as e:
            print("Config Error: Check {ProjectRoot}/config/config.json: %s", str(e))
            try:
                UID = os.environ['UID']
                PSWD = os.environ['PSWD']
            except:
                UID = input("Enter UID:")
                PSWD = input("Enter Password:")

            # default
            self.conf = {
                "Projects": PROJECT_ROOT,
                "EGraderRoot": "EssayGrader/",
                "RemoteMongo": "mongodb+srv://"+UID+":"+PSWD+"@cluster0-ccgud.gcp.mongodb.net",
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
