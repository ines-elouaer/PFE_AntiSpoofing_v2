from src.pad_system.input_validator import PADInputValidator


def main():
    validator = PADInputValidator(
        allow_images=False,
        min_video_frames=16,
        min_video_duration_sec=1.0,
        min_width=64,
        min_height=64,
    )

    test_files = [
        r"E:\PFE_AntiSpoofing_v2\data\processed\casia\all_img\color\10_1.avi_100_real.jpg",

        r"E:\PFE_AntiSpoofing_v2\data\demo\webcam_challenge.mp4",

        r"E:\PFE_AntiSpoofing_v2\fichier_inexistant.mp4",

        
        r"E:\PFE_AntiSpoofing_v2\README.md",
    ]

    for p in test_files:
        print("\n==============================")
        print("FILE:", p)

        result = validator.validate(p)
        print(result.to_dict())


if __name__ == "__main__":
    main()