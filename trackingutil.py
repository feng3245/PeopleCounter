import math
import time
def within_bound(bound, rectangle):
    return bound[0][0] <= rectangle[0][0] and bound[0][1] <= rectangle[0][1] and bound[1][0] >= rectangle[1][0] and bound[1][0] <= rectangle[1][1]
def get_bound(rectangle, bound_ratio):
    bound = ((rectangle[1][0] - rectangle[0][0])*bound_ratio/2, (rectangle[1][1]-rectangle[0][1])*bound_ratio/2)
    return ((rectangle[0][0] - math.ceil(bound[0]), rectangle[0][1] - math.ceil(bound[1])), (rectangle[1][0] + math.ceil(bound[0]), rectangle[1][1] + math.ceil(bound[1])))

class inframe_tracker:
    def __init__(self, seconds_till_oof, bound_ratio):
        self.bound = None
        self.bound_ratio = bound_ratio
        self.timesInframe = []
        self.lastInframe = None
        self.seconds_till_oof = seconds_till_oof
        return
    def track(self, rectangle):
        if not rectangle:
            if self.lastInframe and time.time() - self.lastInframe > self.seconds_till_oof:
                self.timesInframe.append(time.time() - self.lastInframe)
            return
        if not self.bound:
            self.bound = get_bound(rectangle, self.bound_ratio)
            self.lastInframe = time.time()
            return
        if not within_bound(self.bound, rectangle):
            self.timesInframe.append(time.time() - self.lastInframe)
        return