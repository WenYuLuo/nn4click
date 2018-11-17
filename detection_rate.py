

class marked_pool(object):
    def __init__(self, marked_list):
        self.marked_list = marked_list
        self.num_click = len(marked_list)
        self.isdetected = [0 for i in range(self.num_click)]

    def calcu_detection_rate(self, detected_list):
        false_positive = 0
        for position in detected_list:
            position_start = position[0]
            position_end = position[1]
            for i in range(self.num_click):
                marked_start = self.marked_list[i][0]
                marked_end = self.marked_list[i][1]
                a = max(position_start, marked_start)
                b = min(position_end, marked_end)
                if a < b:
                    overlap_len = b-a
                    overlap_rate = overlap_len/(marked_end-marked_start)
                    if overlap_rate >= 0.1:
                        self.isdetected[i] = 1
                        break
                if i == self.num_click-1:
                    false_positive += 1
        correct_detected = [1 for i in self.isdetected if i == 1]
        len_detected = len(correct_detected)
        recall_rate = len_detected/self.num_click
        precision_rate = len_detected/(len_detected+false_positive)
        return recall_rate, precision_rate
