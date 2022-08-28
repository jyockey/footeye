from footeye.app.model import VidInfo, Project, FrameInfo, MatchData

import argparse
import enum
import pickle
import scene_breaker
import ui
import footeye.visionlib.features as features
import footeye.visionlib.frameutils as frameutils
import footeye.utils.framedebug as framedebug


class RunAction(enum.Enum):
    PLAYEREXTRACT = 1
    COLORPICK = 2
    PLAY_PLAYEREXTRACT = 3
    PLAY_MASKTOFIELD = 4
    LINES = 5
    PLAY_LINES = 6
    PITCH_ORIENTATION = 7
    PROCESS_FRAME = 8
    SCENE_BREAK = 9
    PLAY_SCENE = 10

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return RunAction[s.upper()]
        except KeyError:
            return s


def load_project(projectFile):
    with open(projectFile, 'rb') as f:
        return pickle.load(f)


def create_project_from_video(videoFile):
    vid = VidInfo(videoFile)
    projName = input("Project Name?> ")
    project = Project(projName, vid)
    print('Saving to ' + project.filename())
    project.save()
    return project


def process_project(project, action, frame_idx, args):
    vid = project.vidinfo
    if (vid.fieldColorExtents is None):
        vid.fieldColorExtents = features.find_field_color_extents(vid)
        project.save()

    frame = frameutils.extract_frame(vid.vidFilePath, frame_idx)

    if (action == RunAction.COLORPICK):
        features.colorpick_frame(frame)
    elif (action == RunAction.PLAYEREXTRACT):
        framedebug.enable_logging()
        features.extract_players(frame, vid)
        framedebug.show_frames()
    elif (action == RunAction.LINES):
        framedebug.enable_logging()
        features.find_lines(frame, vid)
        framedebug.show_frames()
    elif (action == RunAction.PITCH_ORIENTATION):
        framedebug.enable_logging()
        features.pitch_orientation(frame, vid)
        framedebug.show_frames()
    elif (action == RunAction.PROCESS_FRAME):
        framedebug.enable_logging()
        frame = process_frame(frame, frame_idx, vid)
        framedebug.show_frames()
        print(frame)
    elif (action == RunAction.PLAY_PLAYEREXTRACT):
        ui.play(project, lambda f: features.extract_players(f, vid))
    elif (action == RunAction.PLAY_MASKTOFIELD):
        ui.play(project, lambda f: features.mask_to_field(f, vid))
    elif (action == RunAction.PLAY_LINES):
        ui.play(project, lambda f: features.find_lines(f, vid))
    elif (action == RunAction.SCENE_BREAK):
        project.scenes = scene_breaker.break_scenes(project)
        project.save()
    elif (action == RunAction.PLAY_SCENE):
        ui.play(project, scene=project.scenes[int(args[0])])
    else:
        raise 'UnsupportedAction'


def process_frame(frame, frame_idx, vid):
    frame_info = FrameInfo(frame_idx)
    frame_info.raw_features = features.extract_feature_rects(frame, vid)
    frame_info.player_size = features.estimate_player_size(
            frame_info.raw_features)
    frame_info.player_entities = features.extract_players_from_rects(
            frame, frame_info.raw_features, frame_info.player_size)
    return frame_info


def run_app():
    argparser = argparse.ArgumentParser(description='Run the footeye app')
    group = argparser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p',
                       metavar='PROJECT_FILE',
                       help='an existing project file to open')
    group.add_argument('-v',
                       metavar='VIDEO_FILE',
                       help='a video with which to create a new project')
    argparser.add_argument('-a',
                           metavar='ACTION',
                           help='the action to perform',
                           type=RunAction.argparse,
                           choices=list(RunAction),
                           default='playerextract')
    argparser.add_argument('-f',
                           metavar='FRAME_INDEX',
                           help='the index of a frame to operate on',
                           type=int,
                           default=1)
    argparser.add_argument('rest', nargs=argparse.REMAINDER)
    args = argparser.parse_args()

    if args.p:
        project = load_project(args.p)
    else:
        project = create_project_from_video(args.v)

    process_project(project, args.a, args.f, args.rest)


run_app()
