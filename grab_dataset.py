from pathlib import Path
import shutil
import pandas as pd

DATASET_OUTPUT = Path("/home/bioeos/Documents/Bioeos/aina_dataset_annotation")
SESSIONS = Path("/home/bioeos/Documents/Bioeos/aina_dataset")

def get_code_benthic(code_benthic_filepath: Path) -> dict:

    if not code_benthic_filepath.exists() or not code_benthic_filepath.is_file():
        print(f"{code_benthic_filepath} not found")
        return {}

    codes = {}
    with open(code_benthic_filepath) as file:
        for i, row in enumerate(file):
            if i < 15: continue # Avoid header
            if "Notes" in row: break # Break on footer
            row = row.replace("\n", "").replace('"', '').split(',')
            codes[row[0]] = row[1]

    return codes

def main():
    

    codes = list(get_code_benthic(Path("amoros/cpce_codes_mada_40.txt")).values())
    output_anno = Path(DATASET_OUTPUT, "annotations")
    output_images = Path(DATASET_OUTPUT, "images")
    df_anno_path = Path(DATASET_OUTPUT, "20250410_120000__yves-amoros-mitondrasoa__madagascar_annotation.csv")
    df_anno = pd.DataFrame()
    output_images.mkdir(exist_ok=True, parents=True)
    output_anno.mkdir(exist_ok=True, parents=True)
    for session in sorted(list(SESSIONS.iterdir())):
        dcim_folder = Path(session, "DCIM")

        dict_anno = {}
        metadata_path = Path(session, "METADATA", "metadata.csv")
        metadata_df = pd.read_csv(metadata_path)
        classe = list(set(codes) & set(list(metadata_df)))
        for i, row in metadata_df.iterrows():
            img = Path(session, dcim_folder, row["FileName"])
            cpce_file = Path(session, "PROCESSED_DATA/CPCE_ANNOTATION", f"{img.stem}.cpc")
            
            
            dict_anno[row["FileName"]] = [a for a in classe if row[a] > 0]
            shutil.copy(img, Path(output_images, img.name))
            shutil.copy(cpce_file, Path(output_anno, cpce_file.name))
            
  
        df = pd.DataFrame.from_dict(dict_anno, orient='index')
        # Création des colonnes binaires
        df_binary = pd.DataFrame(0, index=df.index, columns=codes)

        for img, labels in dict_anno.items():
            for label in labels:
                df_binary.loc[img, label] = 1
        df_binary = df_binary.reset_index().rename(columns={'index': 'FileName'})
        df_anno = pd.concat([df_anno, df_binary])
    
    df_anno.to_csv(df_anno_path, index=False)




if __name__ == "__main__":
    main()


