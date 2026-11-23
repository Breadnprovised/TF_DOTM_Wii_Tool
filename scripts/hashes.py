import zlib
def fnv1(s):
    h=0x811c9dc5
    for c in s: h=(h*0x01000193)&0xffffffff; h^=c
    return h
def fnv1a(s):
    h=0x811c9dc5
    for c in s: h^=c; h=(h*0x01000193)&0xffffffff
    return h
def djb2(s):
    h=5381
    for c in s: h=(h*33+c)&0xffffffff
    return h
def djb2x(s):
    h=5381
    for c in s: h=((h*33)^c)&0xffffffff
    return h
def sdbm(s):
    h=0
    for c in s: h=(c+(h<<6)+(h<<16)-h)&0xffffffff
    return h
def x31(s):
    h=0
    for c in s: h=(h*31+c)&0xffffffff
    return h
def x65599(s):
    h=0
    for c in s: h=(h*65599+c)&0xffffffff
    return h
def elf(s):
    h=0
    for c in s:
        h=((h<<4)+c)&0xffffffff; g=h&0xf0000000
        if g: h^=g>>24
        h&=~g&0xffffffff
    return h
def crc(s): return zlib.crc32(s)
def crcn(s): return zlib.crc32(s)^0xffffffff
def jenk(s):
    h=0
    for c in s:
        h=(h+c)&0xffffffff; h=(h+(h<<10))&0xffffffff; h^=h>>6
    h=(h+(h<<3))&0xffffffff; h^=h>>11; h=(h+(h<<15))&0xffffffff
    return h
FS=dict(fnv1=fnv1,fnv1a=fnv1a,djb2=djb2,djb2x=djb2x,sdbm=sdbm,x31=x31,x65599=x65599,elf=elf,crc=crc,crcn=crcn,jenk=jenk)
