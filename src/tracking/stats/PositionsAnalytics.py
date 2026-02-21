
class PositionsAnalytics:
    def __init__(self, ballPositions: list, width: int, height: int, fps: int, realWidth: float, realHeight: float):
        self.ballPositions = ballPositions
        self.width = width
        self.height = height
        self.fps = fps
        self.realWidth = realWidth
        self.realHeight = realHeight
        self.ballPosSpeed = []

    def calculateSpeed(self) -> list:
        scaleX = self.realWidth / self.width
        scaleY = self.realHeight / self.height

        speeds = []
        for i in range(1, len(self.ballPositions)):
            t1, p1 = self.ballPositions[i-1]
            t2, p2 = self.ballPositions[i]
            x1, y1 = p1
            x2, y2 = p2
            dx = (x2 - x1) * scaleX
            dy = (y2 - y1) * scaleY
            realDistance = (dx ** 2 + dy ** 2) ** 0.5
            timeElapsed = (t2 - t1) / self.fps
            speed = realDistance / timeElapsed if timeElapsed > 0 else 0
            speeds.append(speed)
            self.ballPosSpeed.append((t2, x2, y2, speed))
        return speeds

    @property
    def getBallPositionsWithSpeed(self):
        return self.ballPosSpeed

    @property
    def getBallPositions(self):
        return [(t, x, y) for t, x, y in self.ballPositions]

