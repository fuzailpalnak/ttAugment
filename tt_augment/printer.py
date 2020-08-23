from sys import stdout


class Printer:
    @staticmethod
    def print(data):
        if type(data) == dict:
            data = Printer.dict_to_string(data)
        stdout.write("\r\033[1;37m>>\x1b[K" + data.__str__())
        stdout.flush()

    @staticmethod
    def print_new_line(data):
        stdout.write("\n")
        stdout.write("\r\033[1;37m>>\x1b[K" + data.__str__())
        stdout.flush()

    @staticmethod
    def dict_to_string(input_dict, separator=", "):
        combined_list = list()
        for key, value in input_dict.items():
            individual = "{} : {}".format(key, value)
            combined_list.append(individual)
        return separator.join(combined_list)