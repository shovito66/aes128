from BitVector import *
import binascii
import time
import numpy as np

Sbox = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
)

InvSbox = (
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
)

Mixer = [
    [BitVector(hexstring="02"), BitVector(hexstring="03"), BitVector(hexstring="01"), BitVector(hexstring="01")],
    [BitVector(hexstring="01"), BitVector(hexstring="02"), BitVector(hexstring="03"), BitVector(hexstring="01")],
    [BitVector(hexstring="01"), BitVector(hexstring="01"), BitVector(hexstring="02"), BitVector(hexstring="03")],
    [BitVector(hexstring="03"), BitVector(hexstring="01"), BitVector(hexstring="01"), BitVector(hexstring="02")]
]

InvMixer = [
    [BitVector(hexstring="0E"), BitVector(hexstring="0B"), BitVector(hexstring="0D"), BitVector(hexstring="09")],
    [BitVector(hexstring="09"), BitVector(hexstring="0E"), BitVector(hexstring="0B"), BitVector(hexstring="0D")],
    [BitVector(hexstring="0D"), BitVector(hexstring="09"), BitVector(hexstring="0E"), BitVector(hexstring="0B")],
    [BitVector(hexstring="0B"), BitVector(hexstring="0D"), BitVector(hexstring="09"), BitVector(hexstring="0E")]
]

AES_modulus = BitVector(bitstring='100011011')


def print_mySbox(mySbox):
    print("========================SELF GENERATED SBOX/INV S-BOX=============================================")
    rowElement = ""
    for index in range(len(mySbox)):
        if index != 0 and index % 16 == 0:
            rowElement = rowElement + '\n'
        rowElement = rowElement + mySbox[index] + " "

    print(rowElement)


def circularLeftShift_N_bit(myBit, n):
    b = myBit.deep_copy()
    for i in range(n):
        b.circular_rot_left()
    return b


def generate_S_box():
    mySbox = []
    for number in range(0, 256):
        # number = 208 #d2
        hexNumber = '{:x}'.format(int(number))
        # print("HEX NUMBER:{}".format(hexNumber))

        # x = BitVector(hexstring=hexNumber)
        x = BitVector(intVal=number, size=8)
        # print(b)
        # int_val = b.intValue()
        # print("b:{}".format(int_val))
        # print(f"num:{number}--->hex:{hexNumber}")

        modulus = BitVector(bitstring='100011011')
        n = 8
        a = BitVector(bitstring=x)
        if number == 0:
            b = BitVector(bitstring='00000000')
        else:
            b = a.gf_MI(modulus, n)
        # b = a.gf_MI(modulus, n)
        lshift_1 = circularLeftShift_N_bit(b, 1)
        lshift_2 = circularLeftShift_N_bit(b, 2)
        lshift_3 = circularLeftShift_N_bit(b, 3)
        lshift_4 = circularLeftShift_N_bit(b, 4)
        constant16_hex = BitVector(hexstring='63')  # 01100011---63
        s = b ^ lshift_1 ^ lshift_2 ^ lshift_3 ^ lshift_4 ^ constant16_hex

        s = '0x' + str(s.get_bitvector_in_hex())

        mySbox.append(s)

        # print(" b:{}\nL1:{}\nL2:{}\nL3:{}\nL4:{}\n".format(b,lshift_1,lshift_2,lshift_3,lshift_4))
        # print(s)
    return tuple(mySbox)


def generate_S_Inv_Box():
    myInvSbox = []
    for number in range(0, 256):
        modulus = BitVector(bitstring='100011011')
        n = 8
        # number = 208 #d2
        # hexNumber = '{:x}'.format(int(number))
        # print("HEX NUMBER:{}".format(hexNumber))

        # s = BitVector(hexstring=hexNumber)
        s = BitVector(intVal=number, size=8)
        # print(b)
        # int_val = b.intValue()
        # print("b:{}".format(int_val))
        # print(f"num:{number}--->hex:{hexNumber}")

        # b = a.gf_MI(modulus, n)
        lshift_1 = circularLeftShift_N_bit(s, 1)
        lshift_3 = circularLeftShift_N_bit(s, 3)
        lshift_6 = circularLeftShift_N_bit(s, 6)
        constant5_hex = BitVector(hexstring='5')
        b = lshift_1 ^ lshift_3 ^ lshift_6 ^ constant5_hex

        if b.intValue() == 0:
            b = BitVector(bitstring='00000000')
        else:
            b = b.gf_MI(modulus, n)

        b = '0x' + str(b.get_bitvector_in_hex())

        myInvSbox.append(b)

    return tuple(myInvSbox)

print("1. Generate S-Box 2.Generate InvS-Box")
choose_sBox = int(input())
if choose_sBox == 1:
    mySbox = generate_S_box()
    print_mySbox(mySbox)
elif choose_sBox == 2:
    invSBox = generate_S_Inv_Box()
    print_mySbox(invSBox)

print("1.Plain text from file\n2.Plain text from pdf \n3.Insert your plain text\n4.Default ")
choose_input = int(input())
if choose_input == 1:
    # -----------------Open a file: file
    print("Enter your text file name(with .txt extension):")
    filename = input()
    file = open(filename, mode='r')
    # read all lines at once
    plainInputText = file.read()
elif choose_input == 2:
    # ------------------Open a PDF
    # creating a pdf file object
    print("Enter your pdf name(with .pdf extension):")
    filename = input()
    file = open(filename, mode='r')
    plainInputText = file.read()
    # with open(filename, 'rb') as f:
    #     content = f.read()
    # plainInputText = str(binascii.hexlify(content), "utf-8")
elif choose_input == 3:
    plainInputText = input()
else:
    plainInputText = "Two One Nine"

print("--------KEY--------")
print("1..Insert your plain key\n2.Default ")
choose_key = int(input())
if choose_key == 1:
    plainKeyText = input()
else:
    plainKeyText = "Thats my Kung Fu"


####---------------Plain Key Sanitizing-----------
if len(plainKeyText)>16:
  plainKeyText = plainKeyText[0:16]
if len(plainKeyText)<16:
  shortNumber = 16 - len(plainKeyText)
  #print(shortNumber)
  paddingCharacter="0"
  # Append paddingCharacter  N(shortNumber) times
  plainKeyText = plainKeyText.ljust(shortNumber + len(plainKeyText), paddingCharacter)

print(f"Key(ASCII): {plainKeyText}")
plainKeyText = plainKeyText.encode('utf-8')

hexText = plainKeyText.hex()
print(f"Key(HEX): {hexText}")


"""# Generate W[1-4] """
def convert_into_keys(hexText):
  myList = []
  for i in range(4):
    myList.append(hexText[i*8:(i+1)*8])
  return myList


"""> # `G Function`"""


def circular_left_shift(input, d):
    # d bit left shift
    # slice string in two parts for left and right
    d = d * 2
    Lfirst = input[0: d]
    Lsecond = input[d:]
    Rfirst = input[0: len(input) - d]
    Rsecond = input[len(input) - d:]

    input = Lsecond + Lfirst
    # print ("Left Rotation : ", input)
    return input


def circular_right_shift(input, d):
    d = d * 2
    Rfirst = input[0: len(input) - d]
    Rsecond = input[len(input) - d:]
    input = Rsecond + Rfirst
    return input


def sub_byte(hexword):
    answer = ""
    for i in range(4):
        b = BitVector(hexstring=hexword[i * 2:(i + 1) * 2])
        int_val = b.intValue()
        s = Sbox[int_val]
        s = BitVector(intVal=s, size=8).get_bitvector_in_hex()
        answer += s
    return answer


def sub_hexBit(hexbit):
    b = BitVector(hexstring=hexbit)
    int_val = b.intValue()
    s = Sbox[int_val]
    return BitVector(intVal=s, size=8).get_bitvector_in_hex()


def inverse_sub_hexBit(hexbit):
    b = BitVector(hexstring=hexbit)
    int_val = b.intValue()
    s = InvSbox[int_val]
    return BitVector(intVal=s, size=8).get_bitvector_in_hex()


# print(sub_byte("20467567"))
# w_last = sub_byte("20467567")
# w_last

# generate_rc_i()  #testing

def adding_round_constant(wLast):
    rcon = ""
    for i in range(4):
        rcon = rcon + rconVector[i]
    # print(rcon)
    # EX-OR with w_last and rcon

    bv_wLast = BitVector(hexstring=wLast)
    rcon_bitVector = BitVector(hexstring=rcon)

    w_new = bv_wLast ^ rcon_bitVector
    w_new = w_new.get_bitvector_in_hex()
    # w_new = hex(w_new.intValue()).replace("0x","")
    generate_rc_i()
    return w_new


def g_fucntion(hexword, d):
    leftShiftedWord = circular_left_shift(hexword, d)
    # print(f"ls {leftShiftedWord}")
    substituedWord = sub_byte(leftShiftedWord)
    # print(f"sb {substituedWord}")
    w_new = adding_round_constant(substituedWord)
    # print(f"w_new {w_new}")
    # print("After g:{}".format(w_new))
    return w_new


def x_or_hex_string(hex1, hex2):
    bv_hex1 = BitVector(hexstring=hex1)
    bv_hex2 = BitVector(hexstring=hex2)

    w_new = bv_hex1 ^ bv_hex2
    return w_new.get_bitvector_in_hex()


rc_i = "01"
rconList = [rc_i, "00", "00", "00"]
rconVector = np.array(rconList)


# print(rconVector)

def generate_rc_i():
    # only applicable for i>=2
    rc_i_bv = BitVector(hexstring=rconVector[0])

    bv02 = BitVector(hexstring="02")

    bv3 = rc_i_bv.gf_multiply_modular(bv02, AES_modulus, 8)
    hex_rc_i = hex(bv3.intValue()).replace("0x", "")
    if len(hex_rc_i) == 1:
        hex_rc_i = "0" + hex_rc_i
    rconVector[0] = hex_rc_i


"""# **`Task-1`**"""
key_scheduling_start = time.time()
allKeyList = []
keyList = convert_into_keys(hexText)
# print(keyList)
allKeyList.append(keyList.copy())  # appending list
# print(f"allKeyList {allKeyList}")
for k in range(10):

    # print(f"round {k+1}")
    w_last = keyList[3]
    w_new = g_fucntion(w_last, 1)
    # print(f"g_val {w_new}")
    newKeyWordList = []
    # newKeyWordList.append(w_new)
    for i in range(0, 4):
        hex1 = keyList[i]
        if i == 0:
            hex2 = w_new
        else:
            hex2 = newKeyWordList[i - 1]
        w_new2 = x_or_hex_string(hex2, hex1)
        newKeyWordList.append(w_new2)
    keyList.clear()
    keyList = newKeyWordList
    # print(keyList)
    allKeyList.append(keyList.copy())
    # print("------------------------------------------")
key_scheduling_end = time.time()
key_scheduling_time = key_scheduling_end-key_scheduling_start
"""# **`Task-2:AES Encryption`**"""

rows, cols = (4, 4)
columnMatrix = [[0 for i in range(cols)] for j in range(rows)]
inputColumnMatrix =  [[0 for i in range(cols)] for j in range(rows)]
stateMatrix =  [[0 for i in range(cols)] for j in range(rows)]
resultMatrix = [[0 for i in range(cols)] for j in range(rows)]


def generate_column_matrix(extractKey, matrix):
    # print(extractKey)
    row = 0
    col = 0
    for testKey in extractKey:
        # print(testKey)
        myList = []

        for i in range(4):
            # myList.append()
            # print(testKey[i*2:(i+1)*2])
            matrix[row][col] = testKey[i * 2:(i + 1) * 2]
            row = row + 1
        row = 0
        col = col + 1
    return matrix


def add_roundKey(statMatrix):
    for row in range(4):
        for col in range(4):
            hex1 = columnMatrix[row][col]
            hex2 = inputColumnMatrix[row][col]
            statMatrix[row][col] = x_or_hex_string(hex1, hex2)
    return statMatrix


def substitutive_bytes(statMatrix):
    for row in range(4):
        for col in range(4):
            statMatrix[row][col] = sub_hexBit(statMatrix[row][col])
    return statMatrix


def inverse_substitutive_bytes(statMatrix):
    for row in range(4):
        for col in range(4):
            statMatrix[row][col] = inverse_sub_hexBit(statMatrix[row][col])
    return statMatrix


# stateMatrix = substitutive_bytes(stateMatrix)
# stateMatrix

def row_wise_left_shift(statMatrix):
    needToShiftWord = ""
    # row = 2
    for row in range(4):
        needToShiftWord = ""
        for col in range(4):
            needToShiftWord += statMatrix[row][col]
            shiftedWord = circular_left_shift(needToShiftWord, row)
        # print(needToShiftWord)
        for i in range(4):
            statMatrix[row][i] = shiftedWord[i * 2:(i + 1) * 2]
    return statMatrix


def inverse_row_wise_left_shift(statMatrix):
    needToShiftWord = ""
    # row = 2
    for row in range(4):
        needToShiftWord = ""
        for col in range(4):
            needToShiftWord += statMatrix[row][col]
            shiftedWord = circular_right_shift(needToShiftWord, row)
        # print(needToShiftWord)
        for i in range(4):
            statMatrix[row][i] = shiftedWord[i * 2:(i + 1) * 2]
    return statMatrix


# row_wise_left_shift(stateMatrix)
# inverse_row_wise_left_shift(stateMatrix)

def mix_columns(statMatrix):
    resultMatrix = [[0 for i in range(4)] for j in range(4)]
    multiplicationList = ['x', 'x', 'x', 'x']
    # iterate through rows of X
    for i in range(len(Mixer)):
        for j in range(len(statMatrix[0])):
            for k in range(len(statMatrix)):
                mixer_bv = BitVector(hexstring=Mixer[i][k].get_bitvector_in_hex())
                state_bv = BitVector(hexstring=statMatrix[k][j])
                result = mixer_bv.gf_multiply_modular(state_bv, AES_modulus, 8)
                multiplicationList[k] = result
            # print(multiplicationList[k].get_bitvector_in_hex())
            myResult = multiplicationList[0] ^ multiplicationList[1] ^ multiplicationList[2] ^ multiplicationList[3]
            resultMatrix[i][j] = myResult.get_bitvector_in_hex()

    return resultMatrix


def inverse_mix_columns(statMatrix):
    resultMatrix = [[0 for i in range(4)] for j in range(4)]
    multiplicationList = ['x', 'x', 'x', 'x']
    # iterate through rows of X
    for i in range(len(InvMixer)):
        for j in range(len(statMatrix[0])):
            for k in range(len(statMatrix)):
                mixer_bv = BitVector(hexstring=InvMixer[i][k].get_bitvector_in_hex())
                state_bv = BitVector(hexstring=statMatrix[k][j])
                result = mixer_bv.gf_multiply_modular(state_bv, AES_modulus, 8)
                multiplicationList[k] = result
            # print(multiplicationList[k].get_bitvector_in_hex())
            myResult = multiplicationList[0] ^ multiplicationList[1] ^ multiplicationList[2] ^ multiplicationList[3]
            resultMatrix[i][j] = myResult.get_bitvector_in_hex()

    return resultMatrix


"""# `Aggregated Code (Encryption)`"""


def convert_matrixToString(matrix):
    row = 0
    col = 0
    cipherText = ""
    for k in range(4):
        for i in range(4):
            cipherText += matrix[row][col]
            row = row + 1
        row = 0
        col = col + 1
    return cipherText


storeEncryptedChunks = []

##----------Input Part:
# ------------------Plain Input Text Modify (CHUNK BY CHUNK)------------------------#
# def AES_Encryption(plainInputText,inputColumnMatrix,columnMatrix,stateMatrix,storeEncryptedChunks)
print(f"Plain Text(ASCII):{plainInputText}")
encryption_start = time.time()
n = 16  # chunk Size
chunks = [plainInputText[i:i + n] for i in range(0, len(plainInputText), n)]
if len(chunks[-1]) < 16:
    shortNumber = 16 - len(chunks[-1])
    paddingCharacter = " "
    # Append paddingCharacter  N(shortNumber) times
    chunks[-1] = chunks[-1].ljust(shortNumber + len(chunks[-1]), paddingCharacter)
# print(chunks[-1])

# print(f"CHUNKS:{chunks}")

for chunkIndex in range(len(chunks)):

    ##----------Special Round:0
    extractKey = allKeyList[0]
    columnMatrix = generate_column_matrix(extractKey, columnMatrix)

    plainInputText = chunks[chunkIndex].encode('utf-8')
    inputHexText = plainInputText.hex()
    inputHexTextKeys = convert_into_keys(inputHexText)
    # print(inputHexTextKeys)
    inputColumnMatrix = generate_column_matrix(inputHexTextKeys, inputColumnMatrix)

    # ------add round for round 0
    stateMatrix = add_roundKey(stateMatrix)

    # -------------10 times Loop
    for i in range(1, 11):
        # 1st loop ---1-10
        stateMatrix = substitutive_bytes(stateMatrix)
        stateMatrix = row_wise_left_shift(stateMatrix)
        # print(stateMatrix)

        # mix column not valid for 10th loop
        if i != 10:
            stateMatrix = mix_columns(stateMatrix)

        # now stateMatrix--> inputMatrix
        inputColumnMatrix = stateMatrix
        extractKey = allKeyList[i]
        columnMatrix = generate_column_matrix(extractKey, columnMatrix)
        stateMatrix = add_roundKey(stateMatrix)

    storeEncryptedChunks.append(convert_matrixToString(stateMatrix))

encryption_end = time.time()
encryption_time = encryption_end-encryption_start

def convertHexToASCII(hexList):
    concatedHex = ""
    concatedHex = concatedHex.join(hexList)
    return bytes.fromhex(concatedHex).decode('utf-8')


def convertListToString(hexList):
    concatedHex = ""
    return concatedHex.join(hexList)


print(f"\nCiphered Text(Hex) : \n{convertListToString(storeEncryptedChunks)}")

# print(f"Ciphered Text(ASCII) : {convertHexToASCII(storeEncryptedChunks)}")


"""# **`Decryption`**"""
decryption_start = time.time()

storeDecipheredChunks = []
for chunkIndex in range(len(chunks)):

    ##----------Special Round:0
    cipherText = storeEncryptedChunks[chunkIndex]
    extractKey = allKeyList[10]
    columnMatrix = generate_column_matrix(extractKey, columnMatrix)
    cipherTextKeys = convert_into_keys(cipherText)

    # inputColumnMatrix = stateMatrix
    inputColumnMatrix = generate_column_matrix(cipherTextKeys, inputColumnMatrix)

    # stateMatrix =  [[0 for i in range(cols)] for j in range(rows)]
    stateMatrix = add_roundKey(stateMatrix)

    # -------------10 times Loop
    for i in range(9, -1, -1):
        # 1st loop ---1-10

        stateMatrix = inverse_row_wise_left_shift(stateMatrix)
        stateMatrix = inverse_substitutive_bytes(stateMatrix)  # need to change

        # add_round_key
        # now stateMatrix--> inputMatrix
        inputColumnMatrix = stateMatrix
        extractKey = allKeyList[i]
        columnMatrix = generate_column_matrix(extractKey, columnMatrix)
        stateMatrix = add_roundKey(stateMatrix)

        # inverse-mix column not valid for 0-th loop
        if i != 0:
            stateMatrix = inverse_mix_columns(stateMatrix)

        # print(stateMatrix)

    # stateMatrix
    storeDecipheredChunks.append(convert_matrixToString(stateMatrix))
decryption_end = time.time()
decryption_time = decryption_end - decryption_start
print(f"\nDeciphered Text(HEX) : {convertListToString(storeDecipheredChunks)}")
print(f"Deciphered Text(ASCII) : {convertHexToASCII(storeDecipheredChunks)}")

print("\nkey_scheduling_time\t:{}\nEncryption_time\t:{}\ndecryption_time\t:{}".format(key_scheduling_time,encryption_time,decryption_time))

wrtieInAfile = convertHexToASCII(storeDecipheredChunks)
if choose_input==1:
    f = open("Deciphered.txt", "w")
    f.write(wrtieInAfile)
    f.close()
    file.close()
    print("SUCESSFULLY SAVED IN A TXT FILE NAME 'Deciphered.txt'")
elif choose_input == 2:
    f = open("Deciphered.pdf", "w")
    f.write(wrtieInAfile)
    f.close()
    file.close()
    print("SUCESSFULLY SAVED IN A pdf FILE NAME 'Deciphered.pdf'")
