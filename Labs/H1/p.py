
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.fft import dctn, idctn
import cv2


RGB_TO_YCBCR = np.array(
    [[0.299, 0.587, 0.114],
     [-0.168736, -0.331264, 0.5],
     [0.5, -0.418688, -0.081312]]
).T

YCBCR_TO_RGB = np.linalg.inv(RGB_TO_YCBCR)


Q_LUMA = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 28, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

Q_CHROMA = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ]
)


def Q_quality(Q, qualoty):
    """
    Ajusteaza matricea Q in functie de un factor de calitate.
    """
    S = 5000 / qualoty if qualoty < 50 else 200 - 2 * qualoty
    new_Q = np.floor((S * Q + 50) / 100).astype(np.uint16)
    new_Q[new_Q == 0] = 1
    return new_Q


class MyJpeg:
    """
    Clasa simpla pentru compresie JPEG
    """

    def __init__(self, quality=50, blk_size=8):
        """
        Initializeaza parametrii
        """
        self.quality = quality
        self.blk_size = blk_size
        self.Qy = Q_quality(Q_LUMA, quality)
        self.Qc = Q_quality(Q_CHROMA, quality)

    def mse(self, img1, img2):
        """
        Calculeaza eroarea medie patratica
        """
        return np.mean((img1.astype(np.float64) - img2.astype(np.float64))**2)

    def pad_img(self, img):
        """
        Completeaza marginile imaginii
        """
        h, w = img.shape[:2]
        ph = (self.blk_size - (h % self.blk_size)) % self.blk_size
        pw = (self.blk_size - (w % self.blk_size)) % self.blk_size

        if img.ndim == 2:
            pad_img = np.pad(img, ((0, ph), (0, pw)), 'constant')
        else:
            pad_img = np.pad(img, ((0, ph), (0, pw), (0, 0)), 'constant')
        return pad_img, (h, w)

    def unpad_img(self, img, shape):
        """
        Elimina pad-ul pentru a reveni la dimensiunea initiala
        """
        h, w = shape
        if img.ndim == 2:
            return img[:h, :w]
        else:
            return img[:h, :w, :]

    def dct_blk(self, blk):
        """
        Aplica DCT pe bloc
        """
        if blk.ndim == 2:
            return dctn(blk, type=2)
        out = np.zeros_like(blk, dtype=np.float64)
        for i in range(blk.shape[2]):
            out[..., i] = dctn(blk[..., i], type=2)
        return out

    def idct_blk(self, blk):
        """
        Aplica IDCT pe bloc
        """
        if blk.ndim == 2:
            return idctn(blk, type=2)
        out = np.zeros_like(blk, dtype=np.float64)
        for i in range(blk.shape[2]):
            out[..., i] = idctn(blk[..., i], type=2)
        return out

    def compress_gray(self, gray):
        """
        1) Aplica compresia JPEG pe o imagine grayscale
        """
        pad_gray, orig_shape = self.pad_img(gray)
        H, W = pad_gray.shape

        dct_coefs = np.zeros_like(pad_gray, dtype=np.float64)

        for r in range(0, H, self.blk_size):
            for c in range(0, W, self.blk_size):
                block = pad_gray[r:r+self.blk_size, c:c+self.blk_size]
                dct_b = self.dct_blk(block)
                q_b = np.round(dct_b / self.Qy) * self.Qy
                dct_coefs[r:r+self.blk_size, c:c+self.blk_size] = q_b
        
        return dct_coefs, orig_shape

    def decompress_gray(self, dct_coefs, orig_shape):
        """
        Decompresia pentru imagine grayscale
        """
        H, W = dct_coefs.shape
        rec_img = np.zeros_like(dct_coefs, dtype=np.float64)

        for r in range(0, H, self.blk_size):
            for c in range(0, W, self.blk_size):
                block = dct_coefs[r:r+self.blk_size, c:c+self.blk_size]
                rec_b = self.idct_blk(block)
                rec_img[r:r+self.blk_size, c:c+self.blk_size] = rec_b

        rec_img = self.unpad_img(rec_img, orig_shape)
        return np.clip(rec_img, 0, 255).astype(np.uint8)

    def rgb2ycbcr(self, rgb):
        """
        Conversie din RGB in YCbCr
        """
        arr = rgb.astype(np.float32)
        shape = arr.shape
        arr = arr.reshape((-1, 3))

        ycbcr = arr @ RGB_TO_YCBCR
        ycbcr[:, 1:] += 128.0
        return ycbcr.reshape(shape)

    def ycbcr2rgb(self, ycbcr):
        """
        Conversie din YCbCr in RGB
        """
        arr = ycbcr.astype(np.float32)
        shape = arr.shape
        arr = arr.reshape((-1, 3))

        arr[:, 1:] -= 128.0
        rgb = arr @ YCBCR_TO_RGB
        return np.clip(rgb.reshape(shape), 0, 255).astype(np.uint8)

    def compress_color(self, rgb):
        """
        2) Extindere la imagini color
        """
        ycbcr = self.rgb2ycbcr(rgb)
        ycbcr_pad, orig_shape = self.pad_img(ycbcr)
        H, W, _ = ycbcr_pad.shape

        out = np.zeros_like(ycbcr_pad, dtype=np.float64)

        for r in range(0, H, self.blk_size):
            for c in range(0, W, self.blk_size):
                block = ycbcr_pad[r:r+self.blk_size, c:c+self.blk_size, :]
                Y_blk = self.dct_blk(block[..., 0])
                Cb_blk = self.dct_blk(block[..., 1])
                Cr_blk = self.dct_blk(block[..., 2])

                Yq = np.round(Y_blk / self.Qy) * self.Qy
                Cbq = np.round(Cb_blk / self.Qc) * self.Qc
                Crq = np.round(Cr_blk / self.Qc) * self.Qc

                out[r:r+self.blk_size, c:c+self.blk_size, 0] = Yq
                out[r:r+self.blk_size, c:c+self.blk_size, 1] = Cbq
                out[r:r+self.blk_size, c:c+self.blk_size, 2] = Crq

        return out, orig_shape

    def decompress_color(self, data, orig_shape):
        """
        Decompresie pentru imagine color
        """
        H, W, _ = data.shape
        rec = np.zeros_like(data, dtype=np.float64)

        for r in range(0, H, self.blk_size):
            for c in range(0, W, self.blk_size):
                Yq = data[r:r+self.blk_size, c:c+self.blk_size, 0]
                Cbq = data[r:r+self.blk_size, c:c+self.blk_size, 1]
                Crq = data[r:r+self.blk_size, c:c+self.blk_size, 2]

                Y_blk = self.idct_blk(Yq)
                Cb_blk = self.idct_blk(Cbq)
                Cr_blk = self.idct_blk(Crq)

                rec[r:r+self.blk_size, c:c+self.blk_size, 0] = Y_blk
                rec[r:r+self.blk_size, c:c+self.blk_size, 1] = Cb_blk
                rec[r:r+self.blk_size, c:c+self.blk_size, 2] = Cr_blk

        rec = self.unpad_img(rec, orig_shape)
        rgb = self.ycbcr2rgb(rec)
        return rgb

    def compress_until_mse(self, img, mse_th):
        """
        3) compresie pana la un prag MSE dorit
        """
        low = 0.1
        high = 100.0
        bestQ = self.quality
        best_img = img

        for _ in range(10):
            mid = (low + high) / 2.0
            self.Qy = Q_quality(Q_LUMA, mid)
            self.Qc = Q_quality(Q_CHROMA, mid)

            if img.ndim == 2:
                dctcoefs, shp = self.compress_gray(img)
                tmp = self.decompress_gray(dctcoefs, shp)
            else:
                dctcoefs, shp = self.compress_color(img)
                tmp = self.decompress_color(dctcoefs, shp)

            calc_mse = self.mse(img, tmp)

            if calc_mse > mse_th:
                high = mid
            else:
                low = mid
                bestQ = mid
                best_img = tmp

        self.quality = bestQ
        return best_img, bestQ


    def compress_video(self, video_path):
        """
        4) 
        TODO implementare compresie pe fiecare cadru
        """
        pass



if __name__ == "__main__":
    # 1) grayscale
    jpeg = MyJpeg(quality=30)
    X = misc.ascent().astype(np.uint8)

    dcts, sh = jpeg.compress_gray(X)
    Xrec = jpeg.decompress_gray(dcts, sh)

    # MSE
    print("MSE grayscale", jpeg.mse(X, Xrec))
    plt.subplot(1,2,1)
    plt.imshow(X, cmap='gray')
    plt.title("Orig")
    plt.subplot(1,2,2)
    plt.imshow(Xrec, cmap='gray')
    plt.title("JPEG")
    plt.show()


    # 2) color
    img_color = misc.face() 
    jpeg2 = MyJpeg(quality=50)
    cdata, shape_c = jpeg2.compress_color(img_color)
    recon_c = jpeg2.decompress_color(cdata, shape_c)

    print("MSE colr", jpeg2.mse(img_color, recon_c))
    plt.subplot(1,2,1)
    plt.imshow(img_color)
    plt.title("Orig col")
    plt.subplot(1,2,2)
    plt.imshow(recon_c)
    plt.title("JPEG col")
    plt.show()


    # 3) compresie pana la prag
    jpeg3 = MyJpeg()
    reconm, bestQ = jpeg3.compress_until_mse(img_color, mse_th=300.0)
    print("Found best =", bestQ)
    print("mse final =", jpeg3.mse(img_color, reconm))

    # TODO 4) compresie video 