import gzip
import pickle


n = "checkpoints/games/smb1Py/run_10/run-checkpoint-1767.gz";
with gzip.open(n) as f:
    generation, config, population, species_set, rndstate = pickle.load(f);
    print(len(population));
    