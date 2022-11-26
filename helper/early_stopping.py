class EarlyStopping():
    def __init__(self, tolerance=5):

        self.tolerance = tolerance
        self.counter = 0
        self.early_stop = False

    def __call__(self, previous_validation_loss, current_validation_loss):
        if current_validation_loss > previous_validation_loss:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True