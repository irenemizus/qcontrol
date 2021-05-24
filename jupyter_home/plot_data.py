import numpy


def main():
    np = 1024
    nt = 1 #95900
    mod = 10000
    xf = []
    yf = []
    tf = []


    with open("../output/fort.21", 'r') as f_abs:
        for l in range(nt):
            df = read_time_block(f_abs, np)
            if not l % mod:
                xf.append(df[:, 1])
                yf.append(df[:, 2])
                tf.append(df[0, 0])
                print(tf[-1])

    xf = numpy.array(xf)
    yf = numpy.array(yf)

    print(xf)
    print(yf)
    print(tf)


def read_time_block(file, np):
    df = []
    for l in range(np):
        str_arr = file.readline().split()
        flt_arr = []
        for p in range(len(str_arr)):
            flt_arr.append(float(str_arr[p]))
        df.append(flt_arr)

    return numpy.array(df)


if __name__ == "__main__":
    main()