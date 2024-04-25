from datetime import datetime

from schedule.scheduled_movement import ScheduledMovement


class ScheduledService:
    def __init__(self, uid, train_identity, tsc, atoc_code):
        self.movements = []
        self.uid = uid
        self.train_identity = train_identity
        self.tsc = tsc
        self.atoc_code = atoc_code

    def add_stop(self, tiploc: str, time: datetime.time):
        self.movements.append(ScheduledMovement(time, tiploc))

    def get_movements(self) -> [ScheduledMovement]:
        return self.movements

    def get_origin(self) -> ScheduledMovement:
        return self.movements[0]

    def get_destination(self) -> ScheduledMovement:
        return self.movements[len(self.movements)-1]