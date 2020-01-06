import sys
def main():
    # print command line arguments
    arguments = sys.argv[1:]
    if arguments[0] == 'chain_fusion':
        if arguments[1] == 'image' and arguments[2] == 'tabular':
            import chain_fusion_imgtab
        elif arguments[1] == 'tabular':
            print("Err: tabular can't be the first modality of chain fusion.\nTry main [chain_fusion] [image|tabular|text] | [text|tabular|image]")
        else:
            print("main chain_fusion [tabular|image|text] must be given atleast 2 different modalities")
    elif arguments[0] == 'late_fusion':
        if (arguments[1] == 'image' and arguments[2] == 'tabular') or (arguments[1] == 'tabular' and arguments[2] == 'image'):
            import late_fusion_imgtab
    else:
        print("Usage: main [chain_fusion|late_fusion] [image |& text |& tabular]")
if __name__ == "__main__":
    main()