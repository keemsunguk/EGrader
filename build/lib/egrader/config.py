import json
import os
import pathlib

try:
    PROJECT_ROOT = os.environ['PROJECT_ROOT']+'/'
except:
    PROJECT_ROOT = '/'.join(os.environ['PWD'].split('/')[:-2])
    print(PROJECT_ROOT)


class Config:
    """
    Project Configuration
    """

    def __init__(self):
        """
        Load configuration from the project root ./config/config.json
        """
        try:
            with open(pathlib.Path().joinpath(PROJECT_ROOT, 'config', 'config.json'), "r") as confin:
                self.conf = json.load(confin)
        except Exception as e:
            print("Config Error: Check {ProjectRoot}/config/config.json: %s", str(e))
            try:
                UID = os.environ['UID']
                PSWD = os.environ['PSWD']
            except OSError as os_error:
                UID = input("Enter UID:")
                PSWD = input("Enter Password:")

            # default
            self.conf = {
                "Projects": PROJECT_ROOT,
                "EGraderRoot": "EssayGrader/",
                "RemoteMongo": "mongodb+srv://"+UID+":"+PSWD+"@cluster0-ccgud.gcp.mongodb.net",
                "LocalMongo": "mongodb://localhost:27017/",
            }

    def get_config(self) -> json:
        """
        Get configuration
        :return: conf dictionary
        """
        return self.conf

    def __str__(self):
        return str(self.conf)


if __name__ != '__main__':
    config = Config()
