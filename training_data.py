from pathlib import Path
import os
import pickle
from typing import Generic, Iterable, TypeVar

from interrupt import DelayedKeyboardInterrupt

TD = TypeVar('TD');
class TrainingDataManager(Generic[TD]):

    def __init__(self,game_name,run,data_folder="memories",ext="tdat"):
        self.data_file = Path(data_folder)/game_name/(f"run-{run}.{ext}");
        if (os.path.exists(self.data_file)):
            self.load_data();
        else:
            self.next_id:int = 0;
            self.active_data:dict[int,TD] = {};
            self.inactive_data:dict[int,TD] = {};
            self.save_data();

    def __len__(self):
        return len(self.active_data);

    def clear_data(self,save=True):
        for id,datum in self.active_data.items():
            self.inactive_data[id] = datum;
        self.active_data = {};
        if save: self.save_data();

    def add_data(self,data:Iterable[TD],save=True):
        ids_added = [];
        for datum in data:
            self.active_data[self.next_id] = datum;
            ids_added.append(self.next_id);
            self.next_id += 1;
        if save: self.save_data();
        return ids_added;

    def set_data(self,data:Iterable[TD],save=True):
        self.clear_data(save=False);
        return self.add_data(data,save=save);

    def load_data(self):
        with DelayedKeyboardInterrupt():
            with open(self.data_file,'rb') as f:
                ob = pickle.load(f);
                self.next_id = ob['next_id'];
                self.active_data = ob['active_data'];
                self.inactive_data = ob['inactive_data'];

    def save_data(self):
        with DelayedKeyboardInterrupt():
            with open(self.data_file,'wb') as f:
                out = {'next_id':self.next_id, 'active_data':self.active_data, 'inactive_data':self.inactive_data};
                pickle.dump(out,f);

    def get_data_by_id(self,id):
        if id in self.active_data:
            return self.active_data[id];
        elif id in self.inactive_data:
            return self.inactive_data[id];
        else:
            raise IndexError(f"Id {id} not in active or inactive data");

    def __getitem__(self,idx):
        return self.get_data_by_id(idx)
            