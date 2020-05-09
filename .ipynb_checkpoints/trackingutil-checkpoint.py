import math
import time
def within_bound(bound, rectangle):
    return not bound or bound[0][0] <= rectangle[0][0] and bound[0][1] <= rectangle[0][1] and bound[1][0] >= rectangle[1][0] and bound[1][1] >= rectangle[1][1]
def get_bound(rectangle, bound_ratio):
    bound = ((rectangle[1][0] - rectangle[0][0])*bound_ratio/2, (rectangle[1][1]-rectangle[0][1])*bound_ratio/2)
    return ((rectangle[0][0] - math.ceil(bound[0]), rectangle[0][1] - math.ceil(bound[1])), (rectangle[1][0] + math.ceil(bound[0]), rectangle[1][1] + math.ceil(bound[1])))

class object_tracker:
    def __init__(self, seconds_till_oof, bound_ratio):
        self.bound = None
        self.bound_ratio = bound_ratio
        self.firstInframe = None
        self.lastInframe = None
        self.Inframe = True
        self.seconds_till_oof = seconds_till_oof
        return
    def timeInframe(self):
        return self.lastInframe - self.firstInframe
    def track(self, rectangle):
        if not rectangle or (rectangle and not within_bound(self.bound, rectangle)):
            if self.lastInframe and time.time() - self.lastInframe > self.seconds_till_oof:
                self.Inframe = False
                self.lastInframe += self.seconds_till_oof
            return False
        if not self.bound:
            self.bound = get_bound(rectangle, self.bound_ratio)
            self.lastInframe = time.time()
            self.firstInframe = self.lastInframe
            return True
        self.bound = get_bound(rectangle, self.bound_ratio)
        self.lastInframe = time.time()
        return True

class video_tracker:
    def __init__(self, seconds_till_oof, bound_ratio):
        self.active_obs_trackers = []
        self.oof_obs_trackers = []
        self.seconds_till_oof = seconds_till_oof
        self.bound_ratio = bound_ratio
    def track(self, rectangle):
        if not self.active_obs_trackers or not any([o.track(rectangle) for o in self.active_obs_trackers]):
            if rectangle:
                obsTracker = object_tracker(self.seconds_till_oof, self.bound_ratio)
                obsTracker.track(rectangle)
                self.active_obs_trackers.append(obsTracker)
        self.oof_obs_trackers += [o for o in self.active_obs_trackers if not o.Inframe]
        self.active_obs_trackers = [o for o in self.active_obs_trackers if o.Inframe]
        return
    def get_average(self):
        if self.active_obs_trackers + self.oof_obs_trackers:
            return sum([o.timeInframe() for o in (self.active_obs_trackers + self.oof_obs_trackers)]) / self.get_num_objects()
        return 0
    def get_num_current_objects(self):
        return len(self.active_obs_trackers)
    def get_num_objects(self):
        return len(self.active_obs_trackers) + len(self.oof_obs_trackers)