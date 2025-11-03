import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image
import numpy as np

# Clases de madera
CLASS_LABELS = {'BM': 0, 'CM': 1, 'JN': 2, 'HC': 3}
CLASSES = ['BM', 'CM', 'JN', 'HC']

CLASS_DESCRIPTIONS = {
    'BM': 'Faique',
    'CM': 'Cegro', 
    'JN': 'Nogal',  
    'HC': 'Guayacan'  
}


# Umbral mínimo de confianza para considerar una predicción válida
CONFIDENCE_THRESHOLD = 0.8 

# Transformaciones para preprocesamiento. Según la página de Pytorch
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path):
    """
    Carga el modelo PyTorch desde un archivo .pt
    """
    try:
        model = mobilenet_v3_large(weights=None)
        num_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features=num_features, out_features=4)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        # model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None


def predict_image(image, model_path="modelo.pt"):
    """
    Realiza la predicción sobre una imagen
    """
    try:
        # Cargar el modelo
        model = load_model(model_path)
        if model is None:
            return "Error: No se pudo cargar el modelo", {}
        
        # Preprocesar la imagen
        if isinstance(image, str):
            image = Image.open(image)
        
        # Convertir a RGB 
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Aplicar transformaciones
        input_tensor = transform(image).unsqueeze(0)
        
        # Realizar predicción
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # Convertir a numpy para facilitar el manejo
        probs = probabilities.cpu().numpy()
        
        # Crear diccionario con todas las clases y sus probabilidades
        results = {}
        for i, class_name in enumerate(CLASSES):
            display_name = CLASS_DESCRIPTIONS.get(class_name, class_name)
            results[f"{class_name} ({display_name})"] = float(probs[i])
        
        # Encontrar la clase con mayor probabilidad
        max_prob = np.max(probs)
        predicted_class = CLASSES[np.argmax(probs)]
        predicted_display = CLASS_DESCRIPTIONS.get(predicted_class, predicted_class)
        
        # Verificar si supera el umbral de confianza
        if max_prob < CONFIDENCE_THRESHOLD:
            prediction_text = f"🤔 **Madera desconocida**\n\nLa confianza más alta es {max_prob:.2%} para {predicted_class} ({predicted_display}), pero está por debajo del umbral de {CONFIDENCE_THRESHOLD:.2%}"
        else:
            prediction_text = f"🌳 **Clasificación: {predicted_class}** ({predicted_display})\n\nConfianza: {max_prob:.2%}"
        
        return prediction_text, results
        
    except Exception as e:
        error_msg = f"Error durante la predicción: {str(e)}"
        return error_msg, {}
def create_gradio_interface():
    """
    Crea la interfaz de Gradio
    """
    
    # Función wrapper para la interfaz
    def classify_wood(image, model_path, threshold):
        global CONFIDENCE_THRESHOLD
        CONFIDENCE_THRESHOLD = threshold
        
        prediction, probabilities = predict_image(image, model_path)
        
        # Formatear las probabilidades a mostrar
        prob_text = "\n📊 **Probabilidades por clase:**\n"
        for class_name, prob in probabilities.items():
            prob_text += f"• {class_name}: {prob:.2%}\n"
        
        full_result = prediction + "\n" + prob_text
        
        return full_result, probabilities
    
    # Crear la interfaz
    interface = gr.Interface(
        fn=classify_wood,
        inputs=[
            gr.Image(type="pil", label="📸 Subir imagen de madera"),
            gr.Textbox(
                value="best_model_params_RGB.pt", 
                label="📁 Ruta del modelo (.pt)",
                placeholder="Ej: modelo.pt o /ruta/a/tu/modelo.pt"
            ),
            gr.Slider(
                minimum=0.1, 
                maximum=1.0, 
                value=0.5, 
                step=0.05,
                label="🎯 Umbral de confianza",
                info="Probabilidad mínima para considerar una predicción válida"
            )
        ],
        outputs=[
            gr.Textbox(label="🔍 Resultado de la clasificación", lines=10),
            gr.JSON(label="📈 Probabilidades detalladas")
        ],
        title="🌲 Clasificador de Tipos de Madera (BM, CM, JN, HC)",
        description="""
        Sube una imagen de madera y el modelo clasificará el tipo de madera.
        
        **Clases de madera:**
        - **BM**: Tipo BM
        - **CM**: Tipo CM  
        - **JN**: Tipo JN
        - **HC**: Tipo HC
        
        **Características:**
        - Clasifica entre 4 tipos de madera
        - Muestra probabilidades para todas las clases
        - Umbral de confianza configurable
        - Detecta maderas desconocidas
        
        **Instrucciones:**
        1. Sube una imagen clara de la madera
        2. Especifica la ruta de tu modelo .pt
        3. Ajusta el umbral de confianza si es necesario
        4. Haz clic en "Submit" para obtener la clasificación
        """,
        examples=[
        ],
        theme=gr.themes.Soft(),
        flagging_options=None
    )
    
    return interface

# Función principal
def main():
    """
    Función principal para ejecutar la aplicación
    """
    print("🚀 Iniciando aplicación de clasificación de madera...")
    print(f"📋 Clases disponibles: {', '.join(CLASSES)} (BM, CM, JN, HC)")
    print(f"🎯 Umbral de confianza por defecto: {CONFIDENCE_THRESHOLD}")
    
    # Crear y lanzar la interfaz
    interface = create_gradio_interface()
    
    # Lanzar la aplicación
    interface.launch(
        server_name="0.0.0.0",  # Permite acceso desde cualquier IP
        server_port=7860,       # Puerto por defecto de Gradio
        share=False,            # Cambia a True si quieres un enlace público
        debug=True              # Habilita modo debug
    )

if __name__ == "__main__":
    main()

# Versión alternativa para usar en notebook
def launch_notebook():
    """
    Función para lanzar en Jupyter Notebook
    """
    interface = create_gradio_interface()
    return interface.launch(inline=True)

def create_custom_model_interface(model_path, threshold=0.5):
    """
    Crea una interfaz personalizada con parámetros específicos
    """
    global CLASSES, CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = threshold
    
    def classify_custom(image):
        prediction, probabilities = predict_image(image, model_path)
        
        prob_text = "\n📊 **Probabilidades por clase:**\n"
        for class_name, prob in probabilities.items():
            prob_text += f"• {class_name}: {prob:.2%}\n"
        
        return prediction + "\n" + prob_text, probabilities
    
    interface = gr.Interface(
        fn=classify_custom,
        inputs=gr.Image(type="pil", label="📸 Subir imagen de madera"),
        outputs=[
            gr.Textbox(label="🔍 Resultado", lines=8),
            gr.JSON(label="📈 Probabilidades")
        ],
        title="🌲 Clasificador de Madera Personalizado",
        description=f"Modelo: {model_path} | Clases: {', '.join(CLASSES)} | Umbral: {threshold:.2%}"
    )
    
    return interface    


interface = create_custom_model_interface(
    model_path="best_model_params_RGB.pth",
    threshold=0.6
)

    # Lanzar la aplicación
interface.launch(
    server_name="127.0.0.1",  # Permite acceso desde cualquier IP
    server_port=7860,       # Puerto por defecto de Gradio
    share=False,           
    debug=True             
)

if __name__ == "__main__":
    main()