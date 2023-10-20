from QiAlgo_OCR import QiAlgo_OCR


def qi_ml_ocr():
    qi = QiAlgo_OCR()
    # qi = QiAlgo_OCR(
    #     onnx_model_path='PretrainedModel/model.onnx',
    #     model_config_path='PretrainedModel/config.yaml'
    # )

    with open('images/A5Sk.png', 'rb') as f:
        image_bytes = f.read()

    qi.load_image(
        image=image_bytes
    )

    is_successful, results = qi.onnx_predictor()
    if is_successful:
        print(results)  # A5Sk


if __name__ == '__main__':
    qi_ml_ocr()

