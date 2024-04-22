# Ouvrir le fichier en mode lecture
with open("pi_decimals.txt", "r") as file:
    # Lire toutes les lignes
    lines = file.readlines()

# Supprimer les sauts de ligne et les espaces de chaque ligne
decimals = "".join(line.strip() for line in lines)

# Ouvrir un nouveau fichier en mode écriture
with open("pi_decimals_single_line.txt", "w") as file:
    # Écrire les décimales en une seule ligne
    file.write(decimals)