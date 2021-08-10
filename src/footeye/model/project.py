import pickle


class FooteyeProject:
    project_name = None
    vidinfo = None
    home_team = None
    away_team = None

    def __init__(self, project_name):
        self.project_name = project_name

    def save(self):
        filename = self.project_name + ".fae"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
