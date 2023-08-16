import os
import struct
import numpy as np
from scipy.fft import fftn


# A class to handle spectral data processing & to read and write procpar & fid files
class SpectralData:
    """
    Contains methods to process and read spectral data from 'procpar' and 'fid' files.

    Author: Benjamin (Ben) Yoon
    Date: Fri, Jul 21, 2023
    Version: 1.0
    """

    # Static method to read specific lines from a 'procpar' file
    @staticmethod
    def read_write_procpar(read_line, file_path):
        """
        Reads specific lines from the 'procpar' file located at 'file_path'.

        :param read_line: The parameter name to be read from 'procpar'.
        :param file_path: The path to the directory containing the 'procpar' file.
        :return: A list of values associated with the given parameter name.
        :rtype: list of float

        Author: Benjamin (Ben) Yoon
        Date: Fri, Jul 21, 2023
        Version: 1.0
        """
        file_path = file_path + ".fid"
        file_path = os.path.join(file_path, 'procpar')
        with open(file_path) as g:
            read_lines = g.readlines()
            for i, line in enumerate(read_lines):
                if line.strip().startswith(read_line):
                    line_read = read_lines[i + 1].strip()
                    return [float(val) for val in line_read.split()[1:]]

    # Static method to read and process spectral data
    @staticmethod
    def read_write_spectral_data(epsi, file_path, proton_quarter):
        """
        Reads and processes spectral data from the specified file.

        :param epsi: A dictionary containing configuration parameters for data processing.
        :param file_path: The path to the directory containing the spectral data files.
        :param proton_quarter: The proton quarter value used in data processing.
        :return: Processed spectral data as a complex numpy array.
        :rtype: ndarray

        Author: Benjamin (Ben) Yoon
        Date: Fri, Jul 21, 2023
        Version: 1.0
        """
        ne = SpectralData.read_write_procpar('ne', file_path)
        ne = ne[0]
        number_of_points = SpectralData.read_write_procpar('np', file_path)
        number_of_points = number_of_points[0] // 2
        nv = SpectralData.read_write_procpar('nv 1', file_path)
        nv = nv[0]
        te = SpectralData.read_write_procpar('te2', file_path)
        et = 1 / te[0]

        # Arrange echoes
        if epsi['centric']:
            echoes = np.array([0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6]) + 7
        else:
            echoes = np.arange(1, nv + 1)
        tmp_spectral_data_array = np.zeros((int(nv), int(number_of_points), int(ne), epsi['pictures_to_read_write']),
                                           dtype=complex)
        tmp_spectral_data_array_1 = tmp_spectral_data_array.copy()
        tmp_spectral_data_array_2 = tmp_spectral_data_array.copy()
        tmp_spectral_data_array_3 = np.zeros((int(nv), int(number_of_points), int(2 * ne),
                                              epsi['pictures_to_read_write']), dtype=complex)
        spectral_data = tmp_spectral_data_array_3.copy()
        tmp_spectral_data = spectral_data.copy()
        for i in range(epsi['pictures_to_read_write']):
            real_information, imaginary_information, _, _, _, _ = SpectralData.read_write_fid(file_path)
            for j in range(int(ne)):
                ix = np.arange(j, int(nv * ne), int(ne))
                tmp_spectral_data_array[:, :, j, i] = (real_information[:, ix] - 1j * imaginary_information[:, ix]).T
            for j in range(int(nv)):
                tmp_spectral_data_array_1[echoes[j] - 1, :, :, :] = tmp_spectral_data_array[j, :, :, :]
            for j in range(int(nv)):
                for k in range(int(number_of_points)):
                    line = np.squeeze(tmp_spectral_data_array_1[j, k, :, i])
                    tmp_spectral_data_array_2[j, k, :, i] = line * np.exp(
                        -np.arange(0, int(ne)).T * proton_quarter / et)
            tmp_spectral_data_array_3[:, :, 0:int(ne), i] = tmp_spectral_data_array_2[:, :, 0:int(ne), i]
            tmp_spectral_data_1 = np.fft.fftshift(fftn(np.squeeze(tmp_spectral_data_array_3[:, :, :, i])))
            tmp_spectral_data_2 = tmp_spectral_data_1.copy()
            for j in range(0, int(nv)):
                for k in range(0, int(number_of_points)):
                    tmp_spectral_data[j, k, :, i] = tmp_spectral_data_2[j, int(number_of_points - k - 1), ::-1]
        spectral_data = np.abs(tmp_spectral_data)
        return spectral_data

    # Method to read data from a 'fid' file
    def read_write_fid(file_path):
        """
        Reads and processes data from the 'fid' file.

        :param file_path: The path to the 'fid' file.
        :return: A tuple containing various data elements.
        :rtype: tuple

        Author: Benjamin (Ben) Yoon
        Date: Fri, Jul 21, 2023
        Version: 1.0
        """
        path = f"{file_path}.fid/fid"
        with open(path, "rb") as fid:
            blocks = struct.unpack(">i", fid.read(4))[0]
            traces = struct.unpack(">i", fid.read(4))[0]
            points = struct.unpack(">i", fid.read(4))[0]
            eb = struct.unpack(">i", fid.read(4))[0]
            tb = struct.unpack(">i", fid.read(4))[0]
            bb = struct.unpack(">i", fid.read(4))[0]
            vi = struct.unpack(">h", fid.read(2))[0]
            s = struct.unpack(">h", fid.read(2))[0]
            number_of_headers = struct.unpack(">i", fid.read(4))[0]
            s32 = int(bool(s & 4))
            sf = int(bool(s & 8))
            real_information = []
            imaginary_information = []
            b = list(range(1, blocks + 1))
            ob = len(b)
            t = list(range(1, traces + 1))
            ot = len(t)
            i = 1
            j = 1
            for k in range(1, blocks + 1):

                # Read a block header
                scale = struct.unpack(">h", fid.read(2))[0]
                bs = struct.unpack(">h", fid.read(2))[0]
                index = struct.unpack(">h", fid.read(2))[0]
                m = struct.unpack(">h", fid.read(2))[0]
                cc = struct.unpack(">i", fid.read(4))[0]
                lv = struct.unpack(">f", fid.read(4))[0]
                rv = struct.unpack(">f", fid.read(4))[0]
                lvl = struct.unpack(">f", fid.read(4))[0]
                tl = struct.unpack(">f", fid.read(4))[0]
                a = 1
                kk = 0
                for c in range(1, traces + 1):

                    # Read data for each trace
                    if sf == 1:
                        d = struct.unpack(f">{points}f", fid.read(points * 4))
                    elif s32 == 1:
                        d = struct.unpack(f">{points}i", fid.read(points * 4))
                    else:
                        d = struct.unpack(f">{points}h", fid.read(points * 2))

                    # Keep the data if it matches the desired blocks and traces
                    if b[j - 1] == k:
                        if a <= ot:
                            if t[a - 1] == c:
                                real_information.append(list(d[::2]))
                                imaginary_information.append(list(d[1::2]))
                                i += 1
                                a += 1
                                kk = 1
                if kk:
                    j += 1
                if j > ob:
                    break
            real_information = np.array(real_information).T
            imaginary_information = np.array(imaginary_information).T
            number_of_points = points // 2
            number_of_blocks = blocks
            number_of_traces = traces
            header_information = [blocks, traces, points, eb, tb, bb, vi, s, number_of_headers, scale, bs, index, m, cc,
                                  lv, rv, lvl, tl]
            return real_information, imaginary_information, number_of_points, number_of_blocks, number_of_traces, \
                header_information
