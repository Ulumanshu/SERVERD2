import os
import re
import json
import string
import itertools
import random

class Train_Former:
    """ Simple Class witch purpose is to manage filecount
     in train directory, because model accuracy depends on
     even file distribution between data classes
     @staticmethods are used for general funcions,
     File_Copy is the main method to use,
     and accountant creates an info .json file
     TrFo_Self.json"""
     
    def __init__(self,
                 save_dir="../static/Own_classes/save",
                 train_dir="../static/Own_classes/train",
                 json_dir="./",
                 uppercase="/uppercase",
                 lowercase="/lowercase",
                 numbers="/numbers",
                 classifajar="/Classifajar"
                 ):
        self.save_dir = save_dir
        self.train_dir = train_dir
        self.json_dir = json_dir
        self.uppercase = uppercase
        self.lowercase = lowercase
        self.numbers = numbers
        self.classifajar = classifajar
        self.train_upper = self.train_dir + self.uppercase
        self.train_lower = self.train_dir + self.lowercase
        self.train_numbr = self.train_dir + self.numbers
        self.train_class = self.train_dir + self.classifajar
        self.save_upper = self.save_dir + self.uppercase
        self.save_lower = self.save_dir + self.lowercase
        self.save_numbr = self.save_dir + self.numbers
        self.save_list = [self.save_upper, self.save_lower, self.save_numbr]
        self.train_list = [self.train_upper, self.train_lower,
                      self.train_numbr, self.train_class]

    @property
    def Json_Self(self):
        """Loads a content of .json file onto variable
        file is created by accountant method"""
        with open(self.json_dir + "TrFo_Self.json") as f:
            file = json.load(f)
        Json_Self = file
        return Json_Self

    def accountant(self):
        """Counts files in save and train directories
        and creates info .json file TrFo_Self.json"""
        Json_Self = {}
        Json_Self["Save_dir"] = {}
        Json_Self["Train_dir"] = {}
        for e in self.save_list:
            fcs_list = []
            dir_cnt, dir_ls = Train_Former.count_dir(e)
            dir_name_abr = re.search(r"([^\/]+$)", str(e)).group(1)
            Json_Self["Save_dir"][dir_name_abr] = {}
            Json_Self["Save_dir"][dir_name_abr]["Dir_count"] = dir_cnt
            for i in dir_ls:
                file_cnt, file_list = Train_Former.count_file(e + "/" + i)
                fcs_list.append(file_cnt)
                min_value = min(fcs_list)
                total = sum(fcs_list)
                Json_Self["Save_dir"][dir_name_abr]["Min_fc"] = min_value
                Json_Self["Save_dir"][dir_name_abr]["Total_files"] = total
                Json_Self["Save_dir"][dir_name_abr][i] = file_cnt
        for e in self.train_list:
            fct_list = []
            dir_cnt, dir_ls = Train_Former.count_dir(e)
            dir_name_abr = re.search(r"([^\/]+$)", str(e)).group(1)
            Json_Self["Train_dir"][dir_name_abr] = {}
            Json_Self["Train_dir"][dir_name_abr]["Dir_count"] = dir_cnt
            for i in dir_ls:
                file_cnt, file_list = Train_Former.count_file(e + "/" + i)
                fct_list.append(file_cnt)
                min_value = min(fct_list)
                total = sum(fct_list)
                Json_Self["Train_dir"][dir_name_abr]["Min_fc"] = min_value
                Json_Self["Train_dir"][dir_name_abr]["Total_files"] = total
                Json_Self["Train_dir"][dir_name_abr][i] = file_cnt
        with open(self.json_dir + "TrFo_Self.json", 'w') as f:
            json.dump(Json_Self, f, indent=4, sort_keys=True)
    
    @staticmethod
    def sort_key(x):
        """Key function for methods like sort()"""
        key = re.search(r"([^_]+$)", str(x)).group(1)
        key = eval(re.search(r"(^\d+)", str(key)).group(1))
        return key
    
    def Class_former(self):
        """Counts and copies files for
        numbers, upprecase and lowercase classes,
        methods are separate for later and Classifajar
        because they have diferent filetree structure"""
        self.accountant()
        sjson = self.Json_Self
        
        def check_numbering(file_list):
            for index, file_name in enumerate(sorted(file_list, key=Train_Former.sort_key)):
                file_nr = re.search(r"([^_]+$)", str(file_name)).group(1)
                file_nr = eval(re.search(r"(^\d+)", str(file_nr)).group(1))
                if index + 1 != file_nr:
                    return False
                else:
                    continue
            return True
            
        def fix_numbering(save_dir, train_dir):
            save_fileset = set()
            train_fileset = set()
            Train_Former.rename_dir_files(save_dir)
            Train_Former.delete_dir_files(train_dir)
            s_f_count, s_f_list = Train_Former.count_file(save_dir)
            save_fileset.update(s_f_list)
            t_f_count, t_f_list = Train_Former.count_file(train_dir)
            train_fileset.update(t_f_list)
            return save_fileset, train_fileset
            
        for directory in self.save_list:
            dir_name_abr_root = re.search(r"([^\/]+$)", str(directory)).group(1)
            dir_cnt, dir_ls = Train_Former.count_dir(directory)
            min_fc = sjson['Save_dir'][dir_name_abr_root]['Min_fc'] or 0
            for dire in dir_ls:
                save_fileset = set()
                already_copied = set()
                save_dir = os.path.join(directory, dire)
                s_f_count, s_f_list = Train_Former.count_file(save_dir)
                train_dir = os.path.join(self.train_dir + '/' + dir_name_abr_root, dire)
                t_f_count, t_f_list = Train_Former.count_file(train_dir)
                save_fileset.update(s_f_list)
                already_copied.update(t_f_list)
                if len(save_fileset) < len(already_copied):
                    save_fileset, already_copied = fix_numbering(save_dir, train_dir)
                if not check_numbering(list(save_fileset)) or not check_numbering(list(already_copied)):
                    save_fileset, already_copied = fix_numbering(save_dir, train_dir)
                candidates = sorted(list(save_fileset - already_copied), key=Train_Former.sort_key)
                iterations = min_fc - len(already_copied)
                if iterations > 0:
                    for it, fail in zip(range(iterations), candidates):
                        copied_file = Train_Former.read_file(save_dir, fail)
                        Train_Former.save_file(train_dir, fail, copied_file)
    
    def Classifajar_former(self):
        """Counts and copies files for Classifajar class,
            should be used after using Class_former class method.
        """
        s_d = self.Json_Self["Save_dir"]
        cfaj_cnt = min(s_d.items(), key=lambda x: x[1].get("Total_files"))[1].get("Total_files")
        for path in self.save_list:
            already_copied = set()
            dir_name_abr_root = re.search(r"([^\/]+$)", str(path)).group(1)
            dir_cnt, dir_ls = Train_Former.count_dir(path)
            target_dir = self.train_dir + "/Classifajar/" + dir_name_abr_root
            count_target, file_list = Train_Former.count_file(target_dir)
            already_copied.update(file_list)
            itert = cfaj_cnt - count_target
            if itert > 0:
                for it, dir_s in zip(range(itert), itertools.cycle(sorted(dir_ls))):
                    copied = False
                    save_dir = path + "/" + dir_s
                    s_fcount, s_flist = Train_Former.count_file(save_dir)
                    for i, fail in enumerate(sorted(s_flist, key=Train_Former.sort_key)):
                        if fail not in already_copied:
                            copied_file = Train_Former.read_file(save_dir, fail)
                            copied = Train_Former.save_file(target_dir, fail, copied_file)
                            already_copied.add(fail)
                            break
                        if i == len(s_flist) - 1 and not copied:
                            while not copied:
                                random_i = random.randint(0, len(dir_ls) - 1)
                                dir_ss = dir_ls[random_i]
                                save_dirs = path + "/" + dir_ss
                                ss_fcount, ss_flist = Train_Former.count_file(save_dirs)
                                for failss in sorted(ss_flist, key=Train_Former.sort_key):
                                    if failss not in already_copied and not copied:
                                        copied_files = Train_Former.read_file(save_dirs, failss)
                                        copied = Train_Former.save_file(target_dir, failss, copied_files)
                                        already_copied.add(failss)
                                        break
    
    def Purge_Train(self):
        """Deletes all files in train directory"""
        for dir_t in self.train_list:
            dir_cnt, dir_ls = Train_Former.count_dir(dir_t)
            for dir_spec in dir_ls:
                dir_path = os.path.join(dir_t, dir_spec)
                Train_Former.delete_dir_files(dir_path)

    def File_Copy(self):
        """Main class method, its a launcher where methods are called
           in correct order to count and copy files, if you only wish to
           make .json report use accountant method
        """
        self.Class_former()
        self.Classifajar_former()
        return print("Train directory filled.")
    
    @staticmethod
    def delete_dir_files(directory):
        """Deletes files from given directory"""
        file_cnt, file_list = Train_Former.count_file(directory)
        for fail in file_list:
            name = os.path.join(directory, fail)
            os.remove(name)
            print('File %s deleted.' % name)
            
    @staticmethod
    def delete_file(path_to_file):
        """Deletes a single file at given path"""
        os.remove(path_to_file)
        print('File %s deleted.' % path_to_file)
        
    @staticmethod
    def rename_dir_files(directory):
        """Finds files with _number at the end and numbers them again"""
        file_cnt, file_list = Train_Former.count_file(directory)
        for number, fail in enumerate(sorted(file_list), 1):
            file_nr = re.search(r"([^_]+$)", str(fail)).group(1)
            file_nr = re.search(r"(^\d+)", str(file_nr)).group(1)
            fn_root = re.search(r"(^[a-zA-Z]+[_][a-zA-Z0-9]+[_])", str(fail)).group(1)
            new_name = 'c_%s%s.png' % (fn_root, str(number))
            os.rename(directory + '/' + fail, directory + '/' + new_name)
        file_cnt2, file_list2 = Train_Former.count_file(directory)
        for fail2 in sorted(file_list2):
            new_name2 = fail2[2:]
            os.rename(directory + '/' + fail2, directory + '/' + new_name2)

    @staticmethod
    def read_file(dir_c, fname):
        """Reads given file from given destination
        if destination doesnt exist, creates directory"""
        if os.path.exists(dir_c) == False:
            os.makedirs(dir_c)
        with open(os.path.join(dir_c, fname), 'rb') as file:
            output = file.read()
        return output

    @staticmethod
    def save_file(dir_d, fname, file_d):
        """Saves given file with given name to given destination
        if destination doesnt exist, creates directory"""
        if os.path.exists(dir_d) == False:
            os.makedirs(dir_d)
        with open(os.path.join(dir_d, fname), 'wb') as output:
            output.write(file_d)
        return True

    @staticmethod
    def count_dir(dir_d):
        """Counts directories in given destination
        if destination doesnt exist, creates directory,
        returns int(dir_count) and list(dir_list) """
        if os.path.exists(dir_d) == False:
            os.makedirs(dir_d)
        dir_count = len(next(os.walk(dir_d))[1])
        dir_list = []
        for root, dirs, files in os.walk(dir_d):
            dir_list.append(dirs)
        return dir_count, dir_list[0]

    @staticmethod
    def count_file(dir_d):
        """Counts files in given destination
        if destination doesnt exist, creates directory,
        returns int(file_count) and list(file_list) """
        if os.path.exists(dir_d) == False:
            os.makedirs(dir_d)
        file_list = []
        file_count = len(next(os.walk(dir_d))[2])
        for root, dirs, files in os.walk(dir_d):
            file_list.append(files)
        return file_count, file_list[0]

if __name__ == "__main__":
    ozka = Train_Former()
    ozka.Purge_Train()
    
#    ozka.Classifajar_former()
#    ozka.accountant()
    ozka.File_Copy()
#    ozka.Class_former()
