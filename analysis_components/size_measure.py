import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.ticker as ticker


# Equação de interseção de retas
def lineIntersect2(k_x, k_y, l_x, l_y, m_x, m_y, n_x, n_y):
    det = (n_x - m_x) * (l_y - k_y) - (n_y - m_y) * (l_x - k_x)
    if (det == 0.0):
        return 0
    s = ((n_x - m_x) * (m_y - k_y) - (n_y - m_y) * (m_x - k_x)) / det
    t = ((l_x - k_x) * (m_y - k_y) - (l_y - k_y) * (m_x - k_x)) / det
    if (s >= 0 and s <= 1) and (t >= 0 and t <= 1):
        return 1
    else:
        return 0


# Buscando as maiores distancias nos contornos permitidos
def distPixeis(objetos, imgC2):
    controle = -1
    PixeisDistancia = []  # Guarda as informações da linha entre x,y e PontaX,PontaY, como a sua distancia(tamanho) e de qual contorno ela perctence
    ContornosViaveis = []  # Indice de qual contorno dentro de objetos tem area maior que 2000

    for indContorno in range(0,
                             len(objetos)):  # indice de qual contorno estou usando da lista de contornos (objetos) de findContours
        obj = objetos[indContorno]  # Escolho um contorno por vez
        area = cv.contourArea(obj)
        if area > 100 and area < 600000:  # Somente contornos maiores que a área determinada serão considerados
            controle = 0
            ContornosViaveis.append(
                indContorno)  # Guardo o indice da lista de contornos (objetos) do contorno permitido em ContornoViaveis[]
            tam = len(obj)
            for ind in range(0, tam, 3):  # Pego cada pixel que está guardado no contorno
                (x, y) = obj[ind][0]  # Coordenada x,y daquele pixel que faz parte do contorno
                PontaX = 0
                PontaY = 0
                PontaX1_3 = 0
                PontaY1_3 = 0
                PontaX2_3 = 0
                PontaY2_3 = 0
                distancia = 0
                distancia1_3 = 0
                distancia2_3 = 0
                PontaX2_4 = 0
                PontaY2_4 = 0
                distancia2_4 = 0
                PontaX4_6 = 0
                PontaY4_6 = 0
                distancia4_6 = 0
                PontaX5_6 = 0
                PontaY5_6 = 0
                distancia5_6 = 0
                for dist in range(0, tam,
                                  3):  # Busca a distancia do pixel (x,y) para todos os outros (x1,y1) pixeis do contorno
                    (x1, y1) = obj[dist][0]
                    dist_prov = (((x1 - x) ** 2 + (y1 - y) ** 2) ** (1 / 2))
                    if dist_prov > distancia1_3 and dist <= (tam * (1 / 6)):
                        distancia1_3 = dist_prov
                        PontaX1_3 = x1
                        PontaY1_3 = y1
                        continue
                    elif dist_prov > distancia2_4 and dist <= (tam * (2 / 6)):
                        distancia2_4 = dist_prov
                        PontaX2_4 = x1
                        PontaY2_4 = y1
                        continue
                    elif dist_prov > distancia2_3 and dist >= (tam * (3 / 6)):
                        distancia2_3 = dist_prov
                        PontaX2_3 = x1
                        PontaY2_3 = y1
                        continue
                    elif dist_prov > distancia4_6 and dist >= (tam * (4 / 6)):
                        distancia4_6 = dist_prov
                        PontaX4_6 = x1
                        PontaY4_6 = y1
                        continue
                    elif dist_prov > distancia5_6 and dist >= (tam * (5 / 6)):
                        distancia2_3 = dist_prov
                        PontaX5_6 = x1
                        PontaY5_6 = y1
                        continue
                    elif dist_prov > distancia:
                        distancia = dist_prov
                        PontaX = x1
                        PontaY = y1
                        continue
                if (PontaX != 0 and PontaY != 0):
                    PixeisDistancia.append((x, y, PontaX, PontaY, distancia, indContorno))
                if (PontaX1_3 != 0 and PontaY1_3 != 0 and distancia1_3 != 0):
                    PixeisDistancia.append((x, y, PontaX1_3, PontaY1_3, distancia1_3, indContorno))
                if (PontaX2_3 != 0 and PontaY2_3 != 0 and distancia2_3 != 0):
                    PixeisDistancia.append((x, y, PontaX2_3, PontaY2_3, distancia2_3, indContorno))
                if (PontaX2_4 != 0 and PontaY2_4 != 0 and distancia2_4 != 0):
                    PixeisDistancia.append((x, y, PontaX2_4, PontaY2_4, distancia2_4, indContorno))
                if (PontaX4_6 != 0 and PontaY4_6 != 0 and distancia4_6 != 0):
                    PixeisDistancia.append((x, y, PontaX4_6, PontaY4_6, distancia4_6, indContorno))
                if (PontaX5_6 != 0 and PontaY5_6 != 0 and distancia5_6 != 0):
                    PixeisDistancia.append((x, y, PontaX5_6, PontaY5_6, distancia5_6, indContorno))
                cv.drawContours(imgC2, obj, -1, (255, 0, 0), 1)  # (0,0,255)

    PixeisDistancia.sort(key=lambda x: x[4],
                         reverse=True)  # Ordena a lista (PixeisDistancia) em ordem crescente de distancias

    return (PixeisDistancia, ContornosViaveis, imgC2, controle)


def VerifyLine(x, y, PontaX, PontaY, imgBin):
    (Alt, Larg) = imgBin.shape
    matrizZerada = np.zeros((Alt, Larg), np.uint8)
    cv.line(matrizZerada, (x, y), (PontaX, PontaY), (255), 1)
    zeroPoints = cv.findNonZero(matrizZerada)
    contador = 0
    if zeroPoints is None:
        return 0

    for CoordenadasPoint in range(0, len(zeroPoints)):
        (newX, newY) = zeroPoints[CoordenadasPoint][0]
        (a) = imgBin[newY, newX]
        if a > 0:  # 0
            contador = contador + 1

    if (contador < 5):  # 20 #>50
        return 1
    else:
        return 0


# Calcular a maior linha
def calcMaxLine(objetos, ContornosViaveis, PixeisDistancia, reserva):
    IndicePixeisDesenho = []  # Guarda o indice dos par de pixeis que tem a maior distancia de seu contorno que está em PixeisDistancia

    for indContViaveis in range(0, len(ContornosViaveis)):  # Inicia a busca em todos os contornos viáveis
        obj1 = objetos[ContornosViaveis[indContViaveis]]  # Uso o contorno de índice salvo em ContornosViaveis
        for indPixeisDist in range(0,
                                   len(PixeisDistancia)):  # Inicio a busca da maior linha, iniciando pelas maiores distancias
            (x, y, PontaX, PontaY, DistanciaRel, IndiceOp) = PixeisDistancia[indPixeisDist]  # Pegando uma "linha"
            intersec = 0
            if (IndiceOp == ContornosViaveis[
                indContViaveis]):  # Busco a linha que estiver naquele contorno especificado
                for TamContorno in range(0, len(obj1)):
                    (p2x, p2y) = obj1[TamContorno][0]
                    if TamContorno == len(obj1) - 1:
                        (p3x, p3y) = obj1[0][0]
                    else:
                        (p3x, p3y) = obj1[TamContorno + 1][0]
                    intersec = intersec + lineIntersect2(x, y, PontaX, PontaY, p2x, p2y, p3x, p3y)
                    if intersec > 4:
                        break

                if intersec <= 4 and intersec > 0:
                    (x, y, PontaX, PontaY, DistanciaRel, IndiceOp) = PixeisDistancia[indPixeisDist]
                    linha_verificada = VerifyLine(x, y, PontaX, PontaY, reserva)
                    if linha_verificada == 1:
                        continue
                    else:
                        IndicePixeisDesenho.append(indPixeisDist)
                    break  # Dou um break na busca das linhas, e passo para o proximo contorno
    return IndicePixeisDesenho


# Definindo a maior linha
def defineMaxLine(IndicePixeisDesenho, PixeisDistancia, imgC2):
    media = 0
    maiorDistancia = 0
    indiceMaiorDist = 0
    iniX = 0
    iniY = 0
    fimX = 0
    fimY = 0

    for i in range(0, len(IndicePixeisDesenho)):  # Desenhando os Contornos aceitáveis
        (x, y, PontaX, PontaY, DistanciaFinal, _) = PixeisDistancia[IndicePixeisDesenho[i]]
        media = media + DistanciaFinal
        if (DistanciaFinal >= maiorDistancia):
            maiorDistancia = DistanciaFinal
            indiceMaiorDist = i
            iniX = x
            iniY = y
            fimX = PontaX
            fimY = PontaY
        # cv.line(imgC2, (x,y) , (PontaX,PontaY), (255,255,0) , 1 )#(0,255,255)

    '''
    if (len(IndicePixeisDesenho) > 0):#Informando qual a maior zona
        teste = IndicePixeisDesenho[indiceMaiorDist]
        (x,y,PontaX,PontaY,DistanciaFinal,_) = PixeisDistancia[teste]
        cv.line(imgC2, (x,y) , (PontaX,PontaY), (0,0,255) , 1 )#(0,165,255)

        LargCirculo = x
        AltCirculo = y
        tam_micrometro = maiorDistancia * 0.313
        alt_comp = y
        if y > PontaY:
            alt_comp = PontaY
        if alt_comp < AltCirculo: 
            fonte = cv.FONT_HERSHEY_PLAIN#Estilo da fonte a ser escrita
            texto = "{:.2f} micrometros".format(tam_micrometro)
            cv.putText(imgC2,texto,(LargCirculo + 10,AltCirculo - 20),fonte,1,(255,0,255),1,cv.LINE_AA)
        else:
            fonte = cv.FONT_HERSHEY_PLAIN#Estilo da fonte a ser escrita
            texto = "{:.2f} micrometros".format(tam_micrometro)
            cv.putText(imgC2,texto,(LargCirculo + 10,AltCirculo + 20),fonte,1,(255,0,255),1,cv.LINE_AA)
        media = media / len(IndicePixeisDesenho)
        '''
    return (maiorDistancia, iniX, iniY, fimX, fimY)


# Desenhando a maior linha
def drawMaxLine(imgC2, maiorDistancia, x, y, PontaX, PontaY):
    cv.line(imgC2, (x, y), (PontaX, PontaY), (0, 0, 255), 1)  # (0,165,255)
    LargCirculo = x
    AltCirculo = y
    tam_micrometro = maiorDistancia * 0.317  # anterior - 0.313
    alt_comp = y
    if y > PontaY:
        alt_comp = PontaY
    if alt_comp < AltCirculo:
        fonte = cv.FONT_HERSHEY_PLAIN  # Estilo da fonte a ser escrita
        texto = "{:.2f} micra".format(tam_micrometro)
        cv.putText(imgC2, texto, (LargCirculo + 10, AltCirculo - 20), fonte, 0.7, (255, 0, 255), 1, cv.LINE_AA)
    else:
        fonte = cv.FONT_HERSHEY_PLAIN  # Estilo da fonte a ser escrita
        texto = "{:.2f} micra".format(tam_micrometro)
        cv.putText(imgC2, texto, (LargCirculo + 10, AltCirculo + 20), fonte, 0.7, (255, 0, 255), 1, cv.LINE_AA)
    return (imgC2)


# Regua
def Regua(imgC2, zoom, alt, larg):
    proporcaoPixel_Micra = 0.313
    if zoom == 200 and alt == 1532 and larg == 2048:
        proporcaoPixel_Micra = 0.313

    # px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    # fig2,ax = plt.subplots(figsize=(1532*px, 2048*px))#16,22 --- 15.32,20.48
    fig2, ax = plt.subplots(figsize=(12, 12))  # 16,22 --- 15.32,20.48

    fig2.patch.set_facecolor('xkcd:white')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    fig2 = plt.imshow(imgC2)
    scalebar = ScaleBar(proporcaoPixel_Micra, 'um', length_fraction=0.079, sep=3,
                        location='lower right')  # 1 pixel = 0.313 micrometro
    plt.gca().add_artist(scalebar)
    ax.xaxis.tick_top()

    # plt.savefig('Output/imagemFinal_Escala.png', dpi=200,  bbox_inches='tight')# bbox_inches='tight'
    # plt.close()

    return fig2


# ---Fim do desenho da regua

# Tamanho zonas
def tamanho_Zonas(img_bin, contorno):
    maiorDistancia = 0

    imgC2 = img_bin.copy()

    """Listas que vão guardar as informações necessárias"""
    PixeisDistancia = []  # Guarda as informações da linha entre x,y e PontaX,PontaY, como a sua distancia(tamanho) e de qual contorno ela perctence
    ContornosViaveis = []  # Indice de qual contorno dentro de objetos tem area maior que 2000
    IndicePixeisDesenho = []  # Guarda o indice dos par de pixeis que tem a maior distancia de seu contorno que está em PixeisDistancia

    """Criação do vetor de distancias e contornos"""
    PixeisDistancia, ContornosViaveis, imgC2, controle = distPixeis(contorno, imgC2)

    if (controle == -1):
        return 0, img, 0, 0, 0, 0

    """Calculando a maior linha"""
    IndicePixeisDesenho = calcMaxLine(contorno, ContornosViaveis, PixeisDistancia, imgC2)

    """Desenhando a maior linha"""
    maiorDistancia, iniX, iniY, fimX, fimY = defineMaxLine(IndicePixeisDesenho, PixeisDistancia, imgC2)

    return (maiorDistancia, iniX, iniY, fimX, fimY)  # , imgC2


# Main
def start_Medidas(img, tipo):
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # type = 0 -> Cal
    # type = 1 -> Poros
    # type = 2 -> Mg
    # type = 4 -> C2S
    # type = 5 -> C3S

    if tipo == 0:
        label = True
        # label = RegularidadeCal(cv.cvtColor(item,cv.COLOR_RGB2GRAY)).start() #Regularidade do Cal
        # print(label)
        # label = True
    elif tipo == 1:
        # _,bin_img = cv.threshold(cv.cvtColor(item,cv.COLOR_RGB2GRAY),0,255,cv.THRESH_BINARY)
        label = True
        # label = start_RegularidadePoros(bin_img) #Regularidade do Poro
        # rint(label)
    elif tipo == 2:
        # label = labelRegularity(bin_img) #Regularidade do Magnesio
        label = True
    elif tipo == 4:
        # label = labelRegularity(bin_img) #Regularidade do C2S
        label = True
    elif tipo == 5:
        # label = labelRegularity(bin_img) #Regularidade do C2S
        label = True

    if label:
        tam, iniX, iniY, fimX, fimY = tamanho_Zonas(img, contours)
        # Element_reg.append((area,iniX,iniY,fimX,fimY))
    # Element_reg = sorted(Element_reg, reverse=True)
    # Element_reg.sort(key=lambda x: x[0], reverse=True)

    # imagem_final = drawMaxLine(imagem_final,Element_reg[0][0],Element_reg[0][1],Element_reg[0][2],Element_reg[0][3],Element_reg[0][4])
    # imagem_final = drawMaxLine(imagem_final,Element_reg[1][0],Element_reg[1][1],Element_reg[1][2],Element_reg[1][3],Element_reg[1][4])
    """Desenhando uma escala para
     comparacao de tamanho"""
    # Regua(imagem_final, 200, img.shape[0], img.shape[1])

    return iniX, iniY, fimX, fimY, tam
