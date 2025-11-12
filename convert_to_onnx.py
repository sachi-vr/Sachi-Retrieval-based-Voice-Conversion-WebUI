import torch
import sys
import os
import traceback

# Add infer directory to path
now_dir = os.getcwd()
sys.path.append(now_dir)
# inferモジュールへのパスを追加
infer_pack_path = os.path.join(now_dir, "infer", "lib", "infer_pack")
if infer_pack_path not in sys.path:
    sys.path.append(infer_pack_path)

try:
    from models_onnx import SynthesizerTrnMsNSFsidM
except ImportError:
    print("Error: Could not import the necessary model class.")
    print("Please ensure you are running this script from the root of the 'Retrieval-based-Voice-Conversion-WebUI' directory.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

def convert_to_onnx(model_path, exported_path):
    """
    .pth モデルを .onnx 形式に変換します。
    """
    try:
        print(f"モデルを読み込んでいます: {model_path}...")
        cpt = torch.load(model_path, map_location="cpu")
        print("モデルの読み込みに成功しました。")

        # Get config from checkpoint
        config = cpt["config"]
        
        # Determine version string ('v1' or 'v2') based on hidden_channels
        # This logic was previously used to set hidden_channels for dummy input
        # Now we use it to determine the 'version' string for the constructor
        version_string = None
        if 'enc_p.emb_phone.weight' in cpt['weight']:
            weight_dim = cpt['weight']['enc_p.emb_phone.weight'].shape[1]
            if weight_dim == 256:
                version_string = "v1"
            elif weight_dim == 768:
                version_string = "v2"
            else:
                raise ValueError(f"Unknown hidden_channels dimension: {weight_dim}. Cannot determine model version.")
            print(f"重みからバージョンを '{version_string}' と推測します。")
        else:
            # Fallback if 'enc_p.emb_phone.weight' is not directly available
            # This might happen if the model structure is slightly different
            # In this case, we might need to infer from the config list itself
            # For now, let's assume the weight check is robust.
            raise ValueError("モデルのバージョンを特定できませんでした。'enc_p.emb_phone.weight'が見つかりません。")

        # Update number of speakers (as in original export_onnx.py)
        config[-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        print("モデル設定:")
        print(config)

        # Create dummy inputs
        # Use the determined hidden_channels for dummy input
        hidden_channels_for_dummy = 256 if version_string == "v1" else 768
        test_phone = torch.rand(1, 200, hidden_channels_for_dummy)
        test_phone_lengths = torch.tensor([200]).long()
        test_pitch = torch.randint(size=(1, 200), low=5, high=255)
        test_pitchf = torch.rand(1, 200)
        test_ds = torch.LongTensor([0])
        test_rnd = torch.rand(1, 192, 200)

        device = "cpu"

        print("モデルを初期化しています...")
        # Pass 'version' as a keyword argument
        net_g = SynthesizerTrnMsNSFsidM(*config, version=version_string, is_half=False)
        print("モデルの初期化が完了しました。")

        print("重みを読み込んでいます...")
        net_g.load_state_dict(cpt["weight"], strict=False)
        net_g.eval()
        print("State dict loaded.")

        input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
        output_names = ["audio"]

        print(f"モデルをエクスポートしています: {exported_path}...")
        torch.onnx.export(
            net_g,
            (
                test_phone.to(device),
                test_phone_lengths.to(device),
                test_pitch.to(device),
                test_pitchf.to(device),
                test_ds.to(device),
                test_rnd.to(device),
            ),
            exported_path,
            do_constant_folding=False,
            opset_version=16,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
        )
        print("モデルのエクスポートに成功しました。")
        print(f"ONNXモデルが保存されました: {exported_path}")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    MODEL_PATH = "sachirvc2merge1/sachirvc2_80pct_freecool_20pct.pth"
    EXPORTED_PATH = "sachirvc2_80pct_freecool_20pct.onnx"
    
    if not os.path.exists(MODEL_PATH):
        print(f"エラー: モデルファイルが見つかりません: {MODEL_PATH}")
    else:
        convert_to_onnx(MODEL_PATH, EXPORTED_PATH)