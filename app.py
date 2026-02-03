import gradio as gr
import torch
import io
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from TriAttentionArchitectura import WoodClassifierWithTriAttention

# Clases de madera
CLASS_LABELS = {'CM': 0, 'JN': 1, 'BM': 2, 'HC': 3}
CLASSES = ['CM', 'JN', 'BM', 'HC']

CLASS_DESCRIPTIONS = {
    'CM': 'Cedro', 
    'JN': 'Nogal',  
    'BM': 'Faique',
    'HC': 'Guayacan'  
}

# Umbral mínimo de confianza
CONFIDENCE_THRESHOLD = 0.9

# Transformaciones para preprocesamiento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ==================== GRAD-CAM ====================
class GradCAM:
    """Implementación de Grad-CAM para visualizar atención del modelo"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Registrar hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Guarda las activaciones del forward pass"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Guarda los gradientes del backward pass"""
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self):
        """Genera el mapa de calor Grad-CAM"""
        grads = self.gradients
        activations = self.activations
        
        # Global Average Pooling en gradientes → pesos por canal
        weights = grads.mean(dim=(2, 3), keepdim=True)
        
        # Class Activation Map
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Normalizar y redimensionar
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def apply_gradcam(model, img_tensor, device, target_layer):
    """
    Aplica Grad-CAM a una imagen
    
    Returns:
        img: Imagen original normalizada
        heatmap: Mapa de calor
        superimposed: Superposición de heatmap sobre imagen
    """
    model.eval()
    img_tensor_batch = img_tensor.unsqueeze(0).to(device)
    
    # Crear instancia de GradCAM
    gradcam = GradCAM(model, target_layer)
    
    # Forward pass
    output = model(img_tensor_batch)
    class_idx = output.argmax().item()
    
    # Backward pass
    model.zero_grad()
    output[0, class_idx].backward()
    
    # Generar heatmap
    cam = gradcam.generate_heatmap()
    
    # Preparar imagen original
    img = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    
    # Aplicar colormap al heatmap
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    heatmap = heatmap[..., ::-1] / 255.0  # BGR → RGB
    
    # Superposición
    superimposed = 0.4 * heatmap + 0.6 * img
    superimposed = np.clip(superimposed, 0, 1)
    
    return img, heatmap, superimposed


# ==================== FUNCIONES DE CARGA Y PREDICCIÓN ====================
def load_model(model_path):
    """Carga el modelo PyTorch desde un archivo .pt"""
    try:
        model = WoodClassifierWithTriAttention(num_classes=4, use_tri_attention=False)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        
        print("✓ Modelo cargado exitosamente")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None


def predict_image_with_gradcam(image, model_path="modelo.pt", show_gradcam=True):
    """
    Realiza la predicción sobre una imagen con opción de visualizar Grad-CAM
    
    Returns:
        prediction_text: Texto con la predicción
        gradcam_img: Imagen con Grad-CAM (o None)
    """
    try:
        # Cargar el modelo
        model = load_model(model_path)
        if model is None:
            return "Error: No se pudo cargar el modelo", None
        
        # Preprocesar la imagen
        if isinstance(image, str):
            image = Image.open(image)
        
        # Convertir a RGB 
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Guardar imagen original para visualización
        original_image = np.array(image)
        
        # Aplicar transformaciones
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)
        
        # Realizar predicción
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        with torch.no_grad():
            outputs = model(input_batch.to(device))
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Convertir a numpy
        probs = probabilities.cpu().numpy()
        
        # Encontrar la clase con mayor probabilidad
        max_prob = np.max(probs)
        predicted_class = CLASSES[np.argmax(probs)]
        predicted_display = CLASS_DESCRIPTIONS.get(predicted_class, predicted_class)
        
        # Crear texto de predicción con todas las probabilidades
        if max_prob < CONFIDENCE_THRESHOLD:
            prediction_text = f"🤔 **Madera desconocida**\n\n"
            prediction_text += f"La confianza más alta es {max_prob:.2%} para {predicted_class} ({predicted_display}), "
            prediction_text += f"pero está por debajo del umbral de {CONFIDENCE_THRESHOLD:.2%}\n\n"
        else:
            prediction_text = f"🌳 **Clasificación: {predicted_class}** ({predicted_display})\n\n"
            prediction_text += f"Confianza: {max_prob:.2%}\n\n"
        
        # Agregar todas las probabilidades
        prediction_text += "📊 **Probabilidades por clase:**\n"
        for i, class_name in enumerate(CLASSES):
            display_name = CLASS_DESCRIPTIONS.get(class_name, class_name)
            prediction_text += f"• {class_name} ({display_name}): {probs[i]:.2%}\n"
        
        # Generar Grad-CAM si está activado
        gradcam_output = None
        if show_gradcam:
            try:
                # Capa objetivo para Grad-CAM
                target_layer = model.backbone.features[16][0]
                
                # Aplicar Grad-CAM
                img_orig, heatmap, superimposed = apply_gradcam(
                    model, 
                    input_tensor, 
                    device, 
                    target_layer
                )
                
                # Crear visualización combinada
                fig, axes = plt.subplots(1, 1, figsize=(15, 5))
                
                # Superposición
                axes.imshow(superimposed)
                axes.axis('off')
                plt.tight_layout()

                # ✅ CORRECCIÓN: Usar buffer en lugar de tostring_rgb()
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                gradcam_output = np.array(Image.open(buf))
                buf.close()
                plt.close(fig)
                
            except Exception as e:
                print(f"Error generando Grad-CAM: {e}")
                import traceback
                traceback.print_exc()
                prediction_text += f"\n\n⚠️ Error generando Grad-CAM: {str(e)}"
        
        return prediction_text, gradcam_output
        
    except Exception as e:
        import traceback
        error_msg = f"Error durante la predicción: {str(e)}\n{traceback.format_exc()}"
        return error_msg, None
    

# ==================== INTERFAZ GRADIO ====================
def create_custom_model_interface(model_path, threshold=0.9):
    """Crea una interfaz personalizada con parámetros específicos"""
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = threshold
    
    def classify_custom(image, show_gradcam, threshold_value):
        prediction, gradcam_img = predict_image_with_gradcam(
            image, 
            model_path, 
            show_gradcam=show_gradcam
        )
        return prediction, gradcam_img
    

    # ✅ SOLUCIÓN: Sin show_copy_button
    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            f"""
            # 🌲 Clasificador de Tipos de Madera (BM, CM, JN, HC)
            
            Sube una imagen de madera y el modelo clasificará el tipo de madera.

            ---

            ### 🌳 Clases de Madera
            - **CM (Cedro)**
            - **JN (Nogal)**
            - **BM (Faique)**
            - **HC (Guayacán)**
            
            ---
            
            **Características:**
            - ✅ Clasificación automática entre 4 tipos de madera
            - ✅ Muestra probabilidades para todas las clases
            - ✅ Umbral de confianza configurable
            - ✅ **Mapa de calor**: Visualiza qué partes de la imagen usa el modelo
            - ✅ Detecta maderas desconocidas
                
            ---
            
            **Instrucciones:**
            1. Sube una imagen clara de la textura de madera
            2. Especifica la ruta de tu modelo .pt (o usa el predeterminado)
            3. Ajusta el umbral de confianza (0.9 por defecto)
            4. Activa/desactiva el mapa de calor según necesites
            5. Haz clic en "Submit" para obtener la clasificación
         
            **Nota sobre el mapa de calor:**
            - Rojo/Amarillo = Regiones más importantes para la decisión
            - Azul/Verde = Regiones menos relevantes
            
            ---
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="📸 Subir imagen de madera")
                threshold_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=threshold,
                    step=0.05,
                    label="🎯 Umbral de Confianza",
                    info="Probabilidad mínima para considerar válida la predicción"
                )
                gradcam_checkbox = gr.Checkbox(
                    value=True,
                    label="🔥 Mostrar Mapa de calor",
                    info="Visualiza las regiones importantes para la clasificación"
                )
                submit_btn = gr.Button("🚀 Clasificar Madera", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                text_output = gr.Textbox(
                    label="🔍 Resultado de la Clasificación", 
                    lines=12
                    # ✅ Removido: show_copy_button=True
                )
                image_output = gr.Image(
                    label="Mapa de calor", 
                    type="numpy"
                )
        
        # Conectar la función
        submit_btn.click(
            fn=classify_custom,
            inputs=[image_input, gradcam_checkbox, threshold_slider],
            outputs=[text_output, image_output]
        )
    
    return interface

# ==================== EJECUCIÓN ====================
if __name__ == "__main__":
    interface = create_custom_model_interface(
        model_path="Modelo_E12__CON_aumento_sin_tri.pt",
        threshold=0.9
    )
    
    # Lanzar la aplicación
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True
    )