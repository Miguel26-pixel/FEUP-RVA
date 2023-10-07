import cv2

# Carregar a imagem original
imagem_original = cv2.imread('image.jpg')

# Carregar a imagem de substituição
imagem_substituicao = cv2.imread('subs.png')

# Definir a posição e dimensões da ROI na imagem original
x, y, largura, altura = 100, 100, 125, 125

# Redimensionar a imagem de substituição para as dimensões da ROI
imagem_substituicao = cv2.resize(imagem_substituicao, (largura, altura))

# Substituir a ROI pela imagem de substituição
imagem_original[y:y+altura, x:x+largura] = imagem_substituicao

# Salvar a imagem resultante
cv2.imwrite('imagem_alterada.jpg', imagem_original)

# Mostrar a imagem resultante (opcional)
cv2.imshow('Imagem Alterada', imagem_original)
cv2.waitKey(0)
cv2.destroyAllWindows()





