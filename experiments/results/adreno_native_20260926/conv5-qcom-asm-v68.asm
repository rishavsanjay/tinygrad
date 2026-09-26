shps #l52
getone #l52
mov.u32u32 r48.w, 0x8000
mov.u32u32 r49.x, 0x4000
mov.u32u32 r49.y, 864
mov.u32u32 r49.z, 0xfffffe00
mov.u32u32 r49.w, 866
mov.u32u32 r50.x, 868
mov.u32u32 r50.y, 870
mov.u32u32 r50.z, 872
mov.u32u32 r50.w, 874
mov.u32u32 r51.x, 510
mov.u32u32 r51.y, 876
mov.u32u32 r51.z, 512
mov.u32u32 r51.w, 878
mov.u32u32 r52.x, 880
mov.u32u32 r52.y, 8
mov.u32u32 r52.z, 520
mov.u32u32 r52.w, 0xc000
mov.u32u32 r53.x, 0x6000
mov.u32u32 r53.y, 1296
mov.u32u32 r53.z, 648
mov.u32u32 r53.w, 1298
mov.u32u32 r54.x, 649
mov.u32u32 r54.y, 1300
mov.u32u32 r54.z, 650
mov.u32u32 r54.w, 1302
mov.u32u32 r55.x, 651
mov.u32u32 r55.y, 1304
mov.u32u32 r55.z, 652
(ss)nop
stc.u32 c[8], r48.w, 4
(ss)mov.u32u32 r48.w, 1306
mov.u32u32 r49.x, 653
mov.u32u32 r49.y, 1308
mov.u32u32 r49.z, 654
stc.u32 c[12], r49.w, 4
stc.u32 c[16], r50.w, 4
stc.u32 c[20], r51.w, 4
stc.u32 c[24], r52.w, 4
stc.u32 c[28], r53.w, 4
stc.u32 c[32], r54.w, 4
(ss)mov.u32u32 r49.w, 1310
mov.u32u32 r50.x, 655
mov.u32u32 r50.y, 1312
mov.u32u32 r50.z, 656
mov.u32u32 r50.w, 0x2000
stc.u32 c[36], r48.w, 4
(ss)nop
stc.u32 c[40], r49.w, 4
stc.u32 c[44], r50.w, 1
(sy)(ss)shpe
l52:
(jp)shl.b r51.x, r48.x, 6
shl.b r51.z, r48.x, 5
shl.b r51.y, r48.y, 13
mull.u r51.w, r48.z, c2.z
shl.b r52.x, r48.z, 3
mov.u32u32 r52.y, 0
shl.b r0.w, r0.x, 3
shl.b r1.x, r0.y, 9
shl.b r1.y, r0.x, 2
shl.b r1.z, r0.y, 7
mov.u32u32 r2.z, 0
mov.u32u32 r6.y, 0
mov.u32u32 r6.x, 0
mov.u32u32 r5.w, 0
mov.u32u32 r5.z, 0
mov.u32u32 r5.y, 0
mov.u32u32 r5.x, 0
mov.u32u32 r4.w, 0
mov.u32u32 r4.z, 0
mov.u32u32 r4.y, 0
mov.u32u32 r4.x, 0
mov.u32u32 r3.w, 0
mov.u32u32 r3.z, 0
mov.u32u32 r3.y, 0
mov.u32u32 r3.x, 0
mov.u32u32 r2.w, 0
(ss)(rpt2)add.u r1.w, (r)r51.x, (r)r0.w
(rpt1)add.u r0.x, (r)r48.x, (last)(r)r0.x
l80:
(nop3) shl.b r52.z, r52.y, 15
(rpt2)nop
cmps.s.lt p0.x, 0, r0.y
(ss)(nop3) add.u r6.z, r2.x, r52.z
(nop3) sad.s32 r6.w, r51.x, r6.z, r0.w
add.u r7.x, r6.w, -251
add.u r7.y, r6.w, -254
(nop1) add.u r7.z, r6.w, -256
(rpt2)ashr.b r7.w, (r)r7.x, 31
predt
shl.b r8.z, r6.w, 1
shl.b r9.y, r6.w, 1
shl.b r7.w, (last)r7.w, 1
shl.b r8.x, (last)r8.x, 1
add.u r8.z, (last)r8.z, -502
add.u r9.y, (last)r9.y, -508
shrg r7.x, 31, r7.x, (last)r7.w
shrg r7.y, 31, r7.y, (last)r8.x
add.u r8.w, c0.z, (last)r8.z
(nop2) add.u r10.y, c0.z, (last)r9.y
cmps.u.lt hr19.x, r8.w, c0.z
(nop2) cmps.u.lt hr21.z, r10.y, c0.z
cov.u16s32 r9.w, (last)hr19.x
cov.u16s32 r11.x, (last)hr21.z
(rpt1)nop
sad.s32 r9.x, c0.w, r7.x, (last)r9.w
(nop3) sad.s32 r10.z, c0.w, r7.y, (last)r11.x
(rpt1)nop
ldg.u16 hr20.x, g[r8.w], 1
nop
ldg.u16 hr22.z, g[r10.y], 1
predf
(rpt4)nop
mov.u16u16 hr20.x, 0
mov.u16u16 hr22.z, 0
prede
(rpt6)nop
predt
shl.b r6.w, (last)r6.w, 1
(nop2) shl.b r8.y, (last)r8.y, 1
add.u r6.w, (last)r6.w, c2.w
(nop2) shrg r7.z, 31, r7.z, (last)r8.y
(nop3) add.u r11.z, c0.z, (last)r6.w
(nop3) cmps.u.lt hr24.x, r11.z, c0.z
cov.u16s32 r12.y, (last)hr24.x
(rpt2)nop
(nop3) sad.s32 r11.w, c0.w, r7.z, (last)r12.y
(rpt2)nop
ldg.u16 hr25.x, g[r11.z], 1
predf
(rpt4)nop
mov.u16u16 hr25.x, 0
prede
(rpt6)nop
sad.s32 r12.w, r1.w, r2.x, r52.z
(nop2) cmps.s.lt p0.x, 0, r0.x
add.u r13.x, r12.w, -1
add.u r13.y, r12.w, -253
add.u r13.z, r12.w, -255
add.u r13.w, r12.w, 255
(rpt3)ashr.b r14.x, (r)r13.x, 31
predt
shl.b r15.x, r12.w, 1
(nop2) shl.b r14.x, (last)r14.x, 1
add.u r15.x, (last)r15.x, -2
(nop2) shrg r13.x, 31, r13.x, (last)r14.x
(nop3) add.u r15.y, c0.z, (last)r15.x
(nop3) cmps.u.lt hr31.z, r15.y, c0.z
cov.u16s32 r0.z, (last)hr31.z
(rpt2)nop
(nop3) sad.s32 r15.z, c0.w, r13.x, (last)r0.z
(rpt2)nop
ldg.u16 hr2.x, g[r15.y], 1
predf
(rpt4)nop
mov.u16u16 hr2.x, 0
prede
(rpt6)nop
(nop3) cmps.s.lt p0.x, 0, r0.y
(rpt2)nop
predt
shl.b r1.y, r12.w, 1
shl.b r7.y, r12.w, 1
shl.b r14.y, (last)r14.y, 1
shl.b r14.z, (last)r14.z, 1
add.u r1.y, (last)r1.y, -506
add.u r7.y, (last)r7.y, -510
shrg r13.y, 31, r13.y, (last)r14.y
shrg r13.z, 31, r13.z, (last)r14.z
add.u r6.w, c0.z, (last)r1.y
(nop2) add.u r8.y, c0.z, (last)r7.y
cmps.u.lt hr15.x, r6.w, c0.z
(ss)(nop2) cmps.u.lt hr17.z, r8.y, c0.z
cov.u16s32 r7.w, (last)hr15.x
cov.u16s32 r9.x, (last)hr17.z
(rpt1)nop
sad.s32 r7.x, c0.w, r13.y, (last)r7.w
(nop3) sad.s32 r8.z, c0.w, r13.z, (last)r9.x
(rpt1)nop
ldg.u16 hr16.x, g[r6.w], 1
nop
ldg.u16 hr18.z, g[r8.y], 1
predf
(rpt4)nop
mov.u16u16 hr16.x, 0
mov.u16u16 hr18.z, 0
prede
(rpt6)nop
(nop3) cmps.s.lt p0.x, 0, r0.x
(rpt2)nop
predt
shl.b r9.z, r12.w, 1
(nop2) shl.b r14.w, (last)r14.w, 1
(ss)sad.s32 r10.y, c4.y, r9.z, c0.z
(nop2) shrg r13.w, 31, r13.w, (last)r14.w
(nop3) cmps.u.lt hr21.z, r10.y, c0.z
cov.u16s32 r11.x, (last)hr21.z
(rpt2)nop
(nop3) sad.s32 r10.z, c0.w, r13.w, (last)r11.x
(rpt2)nop
ldg.u16 hr22.w, g[r10.y], 1
predf
(rpt4)nop
mov.u16u16 hr22.w, 0
prede
(rpt6)nop
add.u r6.z, (last)r6.z, -257
(nop2) cmps.s.lt p0.y, 0, r0.y
(nop3) add.u r6.z, (last)r6.z, r1.w
(ss)ashr.b r11.z, r6.z, 31
brao !p0.x, !p0.y, #l221
(nop2) sad.s32 r11.w, r6.z, r6.z, c0.z
shl.b r11.z, (last)r11.z, 1
(nop2) cmps.u.lt hr24.z, r11.w, c0.z
shrg r11.z, 31, r6.z, (last)r11.z
cov.u16s32 r13.x, (last)hr24.z
(rpt2)nop
(nop3) sad.s32 r12.x, c0.w, r11.z, (last)r13.x
(rpt2)nop
ldg.u16 hr26.z, g[r11.w], 1
jump #l222
l221:
(jp)mov.u16u16 hr26.z, 0
l222:
(jp)shr.b r52.w, r52.y, 16
mull.u r53.x, r52.y, 9
mull.u r52.w, 9, r52.w
shl.b r52.w, r52.w, 16
add.u r52.w, r52.w, r53.x
add.u r52.w, r51.w, r52.w
add.u r53.z, r52.w, 2
add.u r53.w, r52.w, 216
(rpt1)ashr.b r54.z, (r)r53.z, 31
(rpt1)shl.b r49.x, (r)r54.z, 1
(rpt1)shrg r50.x, 31, (r)r53.z, (r)r49.x
add.u r53.y, r52.w, 1
add.u r54.x, r52.w, 217
shl.b r55.y, r52.w, 1
add.u r54.w, c1.y, r50.y
ashr.b r54.y, r53.y, 31
ashr.b r55.x, r54.x, 31
add.u r49.y, 432, r55.y
shl.b r48.w, r54.y, 1
shl.b r49.z, r55.x, 1
add.u r50.y, r49.y, c1.x
shrg r49.w, 31, r53.y, r48.w
shrg r50.z, 31, r54.x, r49.z
cmps.u.lt hr55.x, r50.y, c1.x
add.u r55.x, c1.y, r50.z
add.u r49.z, 434, r55.y
cov.u16s32 r53.z, hr55.x
add.u r50.z, r49.z, c1.x
add.u r49.y, r54.w, r53.z
cmps.u.lt hr55.y, r50.z, c1.x
cov.u16s32 r53.w, hr55.y
add.u r49.z, r55.x, r53.w
add.u r54.z, c1.y, r50.x
add.u r49.x, 4, r55.y
add.u r54.y, c1.y, r49.w
add.u r48.w, 2, r55.y
add.u r50.x, r49.x, c1.x
add.u r49.w, r48.w, c1.x
cmps.u.lt hr54.w, r50.x, c1.x
cmps.u.lt hr54.z, r49.w, c1.x
cov.u16s32 r53.y, hr54.w
cov.u16s32 r53.x, hr54.z
add.u r49.x, r54.z, r53.y
add.u r48.w, r54.y, r53.x
add.u r50.w, 16, r55.y
add.u r13.z, r12.w, -252
add.u r13.w, r12.w, 256
add.u r14.x, r12.w, 260
add.u r14.y, r12.w, 4
cmps.s.lt p0.x, 0, r0.y
ashr.b r14.z, r13.z, 31
(ss)mov.u32u32 r7.x, r50.y
add.u r50.y, r52.w, 220
ashr.b r53.z, r50.y, 31
shl.b r54.z, r53.z, 1
mov.u32u32 r7.z, r50.z
add.u r50.z, r52.w, 221
ashr.b r53.w, r50.z, 31
shl.b r54.w, r53.w, 1
mov.u32u32 r7.y, r49.y
shrg r49.y, 31, r50.y, r54.z
add.u r53.z, c1.y, r49.y
add.u r54.z, 440, r55.y
add.u r49.y, r54.z, c1.x
cmps.u.lt hr55.x, r49.y, c1.x
cov.u16s32 r54.z, hr55.x
add.u r50.y, r53.z, r54.z
add.u r53.z, r52.w, 224
ashr.b r54.z, r53.z, 31
mov.u32u32 r7.w, r49.z
shrg r49.z, 31, r50.z, r54.w
add.u r53.w, c1.y, r49.z
add.u r54.w, 442, r55.y
add.u r49.z, r54.w, c1.x
cmps.u.lt hr55.y, r49.z, c1.x
cov.u16s32 r54.w, hr55.y
add.u r50.z, r53.w, r54.w
add.u r53.w, r52.w, 3
ashr.b r54.w, r53.w, 31
mov.u32u32 r6.z, r50.x
add.u r50.x, r52.w, 219
ashr.b r53.y, r50.x, 31
shl.b r54.y, r53.y, 1
mov.u32u32 r15.z, r49.w
add.u r49.w, r52.w, 218
ashr.b r53.x, r49.w, 31
shl.b r54.x, r53.x, 1
mov.u32u32 r6.w, r49.x
shrg r49.x, 31, r50.x, r54.y
add.u r53.y, c1.y, r49.x
add.u r54.y, 438, r55.y
add.u r49.x, r54.y, c1.x
cmps.u.lt hr54.w, r49.x, c1.x
cov.u16s32 r54.y, hr54.w
add.u r50.x, r53.y, r54.y
add.u r53.y, r52.w, 223
ashr.b r54.y, r53.y, 31
mov.u32u32 r15.w, r48.w
shrg r48.w, 31, r49.w, r54.x
add.u r53.x, c1.y, r48.w
add.u r54.x, 436, r55.y
add.u r48.w, r54.x, c1.x
(rpt2)ashr.b r14.w, (r)r13.w, 31
(ss)cmps.u.lt hr51.z, r48.w, c1.x
cov.u16s32 r54.x, hr51.z
mov.u32u32 r10.y, r49.y
shl.b r49.y, r54.z, 1
mov.u32u32 r10.z, r50.y
shrg r50.y, 31, r53.z, r49.y
add.u r54.z, c1.y, r50.y
add.u r49.y, 448, r55.y
add.u r50.y, r49.y, c1.x
cmps.u.lt hr55.x, r50.y, c1.x
cov.u16s32 r53.z, hr55.x
add.u r49.y, r54.z, r53.z
mov.u32u32 r10.w, r49.z
shl.b r49.z, r54.w, 1
ldg.u16 hr1.x, g[r15.z], 1
ldg.u16 hr16.y, g[r6.z], 1
ldg.u16 hr16.z, g[r7.x], 1
ldg.u16 hr16.w, g[r7.z], 1
mov.u32u32 r11.x, r50.z
shrg r50.z, 31, r53.w, r49.z
add.u r54.w, c1.y, r50.z
add.u r49.z, 6, r55.y
add.u r50.z, r49.z, c1.x
cmps.u.lt hr55.y, r50.z, c1.x
cov.u16s32 r53.w, hr55.y
add.u r49.z, r54.w, r53.w
mov.u32u32 r9.z, r49.x
shl.b r49.x, r54.y, 1
mov.u32u32 r9.w, r50.x
shrg r50.x, 31, r53.y, r49.x
add.u r54.y, c1.y, r50.x
add.u r49.x, 446, r55.y
add.u r50.x, r49.x, c1.x
cmps.u.lt hr54.w, r50.x, c1.x
cov.u16s32 r53.y, hr54.w
add.u r49.x, r54.y, r53.y
mov.u32u32 r8.z, r48.w
(ss)add.u r49.w, r53.x, r54.x
add.u r53.x, r52.w, 222
ashr.b r54.x, r53.x, 31
shl.b r48.w, r54.x, 1
mov.u32u32 r6.z, r50.y
add.u r50.y, r52.w, 433
ashr.b r53.z, r50.y, 31
shl.b r54.z, r53.z, 1
mov.u32u32 r6.w, r49.y
shrg r49.y, 31, r50.y, r54.z
add.u r53.z, c1.y, r49.y
add.u r54.z, c3.x, r55.y
add.u r49.y, r54.z, c1.x
cmps.u.lt hr55.x, r49.y, c1.x
cov.u16s32 r54.z, hr55.x
add.u r50.y, r53.z, r54.z
add.u r53.z, r52.w, 437
ashr.b r54.z, r53.z, 31
mov.u32u32 r7.x, r50.z
add.u r50.z, r52.w, 434
ashr.b r53.w, r50.z, 31
shl.b r54.w, r53.w, 1
mov.u32u32 r7.y, r49.z
shrg r49.z, 31, r50.z, r54.w
add.u r53.w, c1.y, r49.z
add.u r54.w, c3.y, r55.y
add.u r49.z, r54.w, c1.x
cmps.u.lt hr55.y, r49.z, c1.x
cov.u16s32 r54.w, hr55.y
add.u r50.z, r53.w, r54.w
add.u r53.w, r52.w, 438
ashr.b r54.w, r53.w, 31
mov.u32u32 r15.z, r50.x
add.u r50.x, r52.w, 432
ashr.b r53.y, r50.x, 31
shl.b r54.y, r53.y, 1
mov.u32u32 r15.w, r49.x
shrg r49.x, 31, r50.x, r54.y
add.u r53.y, c1.y, r49.x
add.u r54.y, c2.z, r55.y
add.u r49.x, r54.y, c1.x
cmps.u.lt hr54.w, r49.x, c1.x
cov.u16s32 r54.y, hr54.w
add.u r50.x, r53.y, r54.y
add.u r53.y, r52.w, 436
ashr.b r54.y, r53.y, 31
(ss)mov.u32u32 r8.w, r49.w
shrg r49.w, 31, r53.x, r48.w
add.u r54.x, c1.y, r49.w
add.u r48.w, 444, r55.y
add.u r49.w, r48.w, c1.x
cmps.u.lt hr54.z, r49.w, c1.x
cov.u16s32 r53.x, hr54.z
add.u r48.w, r54.x, r53.x
(ss)mov.u32u32 r12.x, r49.w
add.u r49.w, r52.w, 4
ashr.b r53.x, r49.w, 31
shl.b r54.x, r53.x, 1
mov.u32u32 r12.y, r48.w
shrg r48.w, 31, r49.w, r54.x
add.u r53.x, c1.y, r48.w
add.u r54.x, 8, r55.y
add.u r48.w, r54.x, c1.x
(ss)cmps.u.lt hr51.z, r48.w, c1.x
cov.u16s32 r54.x, hr51.z
(rpt3)nop
ldg.u16 hr23.x, g[r8.z], 1
ldg.u16 hr23.y, g[r9.z], 1
ldg.u16 hr23.z, g[r10.y], 1
ldg.u16 hr23.w, g[r10.w], 1
(ss)add.u r49.w, r53.x, r54.x
add.u r53.x, r52.w, 435
ashr.b r54.x, r53.x, 31
mov.u32u32 r10.y, r49.y
shl.b r49.y, r54.z, 1
mov.u32u32 r10.z, r50.y
shrg r50.y, 31, r53.z, r49.y
add.u r54.z, c1.y, r50.y
add.u r49.y, c4.x, r55.y
add.u r50.y, r49.y, c1.x
cmps.u.lt hr55.x, r50.y, c1.x
cov.u16s32 r53.z, hr55.x
add.u r49.y, r54.z, r53.z
ldg.u16 hr15.x, g[r12.x], 1
ldg.u16 hr15.y, g[r15.z], 1
ldg.u16 hr15.z, g[r6.z], 1
ldg.u16 hr15.w, g[r7.x], 1
mov.u32u32 r10.w, r49.z
shl.b r49.z, r54.w, 1
mov.u32u32 r11.x, r50.z
shrg r50.z, 31, r53.w, r49.z
add.u r54.w, c1.y, r50.z
add.u r49.z, c4.z, r55.y
add.u r50.z, r49.z, c1.x
cmps.u.lt hr55.y, r50.z, c1.x
cov.u16s32 r53.w, hr55.y
add.u r49.z, r54.w, r53.w
mov.u32u32 r8.z, r48.w
shl.b r48.w, r54.x, 1
mov.u32u32 r9.z, r49.x
shl.b r49.x, r54.y, 1
mov.u32u32 r9.w, r50.x
shrg r50.x, 31, r53.y, r49.x
add.u r54.y, c1.y, r50.x
add.u r49.x, c3.w, r55.y
add.u r50.x, r49.x, c1.x
cmps.u.lt hr54.w, r50.x, c1.x
cov.u16s32 r53.y, hr54.w
add.u r49.x, r54.y, r53.y
(ss)mov.u32u32 r8.w, r49.w
shrg r49.w, 31, r53.x, r48.w
add.u r54.x, c1.y, r49.w
add.u r48.w, c3.z, r55.y
add.u r49.w, r48.w, c1.x
cmps.u.lt hr54.z, r49.w, c1.x
cov.u16s32 r53.x, hr54.z
add.u r48.w, r54.x, r53.x
mov.u32u32 r7.x, r50.y
add.u r50.y, r52.w, 5
ashr.b r53.z, r50.y, 31
shl.b r54.z, r53.z, 1
mov.u32u32 r7.y, r49.y
shrg r49.y, 31, r50.y, r54.z
add.u r53.z, c1.y, r49.y
add.u r54.z, 10, r55.y
add.u r49.y, r54.z, c1.x
cmps.u.lt hr55.x, r49.y, c1.x
cov.u16s32 r54.z, hr55.x
add.u r50.y, r53.z, r54.z
add.u r53.z, r52.w, c7.w
ashr.b r54.z, r53.z, 31
mov.u32u32 r6.z, r50.x
add.u r50.x, r52.w, 440
ashr.b r53.y, r50.x, 31
shl.b r54.y, r53.y, 1
mov.u32u32 r6.w, r49.x
shrg r49.x, 31, r50.x, r54.y
add.u r53.y, c1.y, r49.x
add.u r54.y, c5.y, r55.y
add.u r49.x, r54.y, c1.x
cmps.u.lt hr54.w, r49.x, c1.x
cov.u16s32 r54.y, hr54.w
add.u r50.x, r53.y, r54.y
add.u r53.y, r52.w, c7.y
(nop2) ashr.b r54.y, r53.y, 31
ldg.u16 hr24.x, g[r8.z], 1
ldg.u16 hr24.y, g[r9.z], 1
ldg.u16 hr24.z, g[r10.y], 1
ldg.u16 hr24.w, g[r10.w], 1
(ss)mov.u32u32 r8.z, r50.z
add.u r50.z, r52.w, 6
ashr.b r53.w, r50.z, 31
shl.b r54.w, r53.w, 1
mov.u32u32 r8.w, r49.z
shrg r49.z, 31, r50.z, r54.w
add.u r53.w, c1.y, r49.z
add.u r54.w, 12, r55.y
add.u r49.z, r54.w, c1.x
cmps.u.lt hr55.y, r49.z, c1.x
cov.u16s32 r54.w, hr55.y
add.u r50.z, r53.w, r54.w
add.u r53.w, r52.w, c8.y
ashr.b r54.w, r53.w, 31
mov.u32u32 r15.z, r49.w
add.u r49.w, r52.w, 439
ashr.b r53.x, r49.w, 31
shl.b r54.x, r53.x, 1
mov.u32u32 r15.w, r48.w
shrg r48.w, 31, r49.w, r54.x
add.u r53.x, c1.y, r48.w
add.u r54.x, c5.x, r55.y
add.u r48.w, r54.x, c1.x
mov.u32u32 r10.w, r49.x
shl.b r49.x, r54.y, 1
mov.u32u32 r11.x, r50.x
shrg r50.x, 31, r53.y, r49.x
add.u r54.y, c1.y, r50.x
add.u r49.x, c7.x, r55.y
add.u r50.x, r49.x, c1.x
cmps.u.lt hr54.w, r50.x, c1.x
cov.u16s32 r53.y, hr54.w
add.u r49.x, r54.y, r53.y
(ss)cmps.u.lt hr51.z, r48.w, c1.x
cov.u16s32 r54.x, hr51.z
mov.u32u32 r10.y, r48.w
(rpt2)nop
ldg.u16 hr18.x, g[r15.z], 1
ldg.u16 hr18.y, g[r6.z], 1
ldg.u16 hr18.w, g[r7.x], 1
ldg.u16 hr19.x, g[r8.z], 1
(ss)add.u r49.w, r53.x, r54.x
add.u r53.x, r52.w, c6.w
ashr.b r54.x, r53.x, 31
shl.b r48.w, r54.x, 1
mov.u32u32 r15.z, r49.y
shl.b r49.y, r54.z, 1
mov.u32u32 r15.w, r50.y
shrg r50.y, 31, r53.z, r49.y
add.u r54.z, c1.y, r50.y
add.u r49.y, c7.z, r55.y
add.u r50.y, r49.y, c1.x
cmps.u.lt hr55.x, r50.y, c1.x
cov.u16s32 r53.z, hr55.x
add.u r49.y, r54.z, r53.z
mov.u32u32 r6.z, r49.z
shl.b r49.z, r54.w, 1
mov.u32u32 r6.w, r50.z
shrg r50.z, 31, r53.w, r49.z
add.u r54.w, c1.y, r50.z
add.u r49.z, c8.x, r55.y
add.u r50.z, r49.z, c1.x
cmps.u.lt hr55.y, r50.z, c1.x
cov.u16s32 r53.w, hr55.y
add.u r49.z, r54.w, r53.w
(ss)mov.u32u32 r10.z, r49.w
shrg r49.w, 31, r53.x, r48.w
add.u r54.x, c1.y, r49.w
add.u r48.w, c6.z, r55.y
add.u r49.w, r48.w, c1.x
cmps.u.lt hr54.z, r49.w, c1.x
cov.u16s32 r53.x, hr54.z
(nop3) add.u r48.w, r54.x, r53.x
(rpt2)nop
ldg.u16 hr14.x, g[r10.y], 1
ldg.u16 hr14.y, g[r10.w], 1
ldg.u16 hr14.z, g[r15.z], 1
ldg.u16 hr14.w, g[r6.z], 1
(ss)mov.u32u32 r10.w, r50.y
add.u r50.y, r52.w, c9.w
ashr.b r53.z, r50.y, 31
shl.b r54.z, r53.z, 1
mov.u32u32 r11.x, r49.y
shrg r49.y, 31, r50.y, r54.z
add.u r53.z, c1.y, r49.y
add.u r54.z, c9.z, r55.y
add.u r49.y, r54.z, c1.x
cmps.u.lt hr55.x, r49.y, c1.x
cov.u16s32 r54.z, hr55.x
add.u r50.y, r53.z, r54.z
add.u r53.z, r52.w, 8
mov.u32u32 r15.z, r50.z
add.u r50.z, r52.w, c10.y
ashr.b r53.w, r50.z, 31
shl.b r54.w, r53.w, 1
mov.u32u32 r15.w, r49.z
shrg r49.z, 31, r50.z, r54.w
add.u r53.w, c1.y, r49.z
add.u r54.w, c10.x, r55.y
add.u r49.z, r54.w, c1.x
cmps.u.lt hr55.y, r49.z, c1.x
cov.u16s32 r54.w, hr55.y
add.u r50.z, r53.w, r54.w
mov.u32u32 r8.z, r49.w
add.u r49.w, r52.w, c8.w
ashr.b r53.x, r49.w, 31
shl.b r54.x, r53.x, 1
mov.u32u32 r8.w, r48.w
shrg r48.w, 31, r49.w, r54.x
add.u r53.x, c1.y, r48.w
add.u r54.x, c8.z, r55.y
add.u r48.w, r54.x, c1.x
mov.u32u32 r10.y, r50.x
add.u r50.x, r52.w, c9.y
ashr.b r53.y, r50.x, 31
shl.b r54.y, r53.y, 1
mov.u32u32 r10.z, r49.x
shrg r49.x, 31, r50.x, r54.y
add.u r53.y, c1.y, r49.x
add.u r54.y, c9.x, r55.y
add.u r49.x, r54.y, c1.x
cmps.u.lt hr54.w, r49.x, c1.x
cov.u16s32 r54.y, hr54.w
add.u r50.x, r53.y, r54.y
add.u r53.y, r52.w, 7
ashr.b r54.y, r53.z, 31
shl.b r55.x, r54.y, 1
(ss)cmps.u.lt hr51.z, r48.w, c1.x
cov.u16s32 r54.x, hr51.z
mov.u32u32 r6.z, r48.w
(rpt2)nop
ldg.u16 hr1.y, g[r8.z], 1
(rpt1)nop
ldg.u16 hr2.y, g[r10.y], 1
ldg.u16 hr2.z, g[r10.w], 1
ldg.u16 hr2.w, g[r15.z], 1
(ss)add.u r49.w, r53.x, r54.x
add.u r53.x, r52.w, c10.w
ashr.b r54.x, r53.y, 31
ashr.b r53.w, r53.x, 31
shl.b r54.w, r54.x, 1
shl.b r54.z, r53.w, 1
shrg r48.w, 31, r53.x, r54.z
ashr.b r54.z, r52.w, 31
shl.b r54.z, r54.z, 1
shrg r54.z, 31, r52.w, r54.z
add.u r54.z, c1.y, r54.z
add.u r52.w, r52.w, r52.w
add.u r52.w, r52.w, c1.x
cmps.u.lt hr48.x, r52.w, c1.x
mov.u32u32 r8.z, r49.x
shrg r49.x, 31, r53.y, r54.w
mov.u32u32 r10.z, r50.y
add.u r50.y, c10.z, r55.y
add.u r53.x, r50.y, c1.x
mov.u32u32 r11.x, r50.z
add.u r50.z, 14, r55.y
add.u r53.y, r50.z, c1.x
mov.u32u32 r10.y, r49.y
shrg r49.y, 31, r53.z, r55.x
add.u r53.z, r50.w, c1.x
mov.u32u32 r10.w, r49.z
add.u r49.z, c1.y, r48.w
mov.u32u32 r8.w, r50.x
add.u r50.x, c1.y, r49.y
(ss)(rpt1)cmps.u.lt hr49.z, (r)r53.x, c1.x
(rpt1)cov.u16s32 r50.y, (r)hr49.z
add.u r53.w, r49.z, r50.y
mov.u32u32 r6.w, r49.w
add.u r49.w, c1.y, r49.x
add.u r54.x, r49.w, r50.z
cmps.u.lt hr50.x, r53.z, c1.x
cov.u16s32 r50.w, hr50.x
add.u r54.y, r50.x, r50.w
mov.u32u32 r15.z, r53.x
(ss)cov.u16s32 r48.w, hr48.x
add.u r54.z, r54.z, r48.w
mov.u32u32 r15.w, r53.w
(rpt3)nop
ldg.u16 hr25.y, g[r6.z], 1
ldg.u16 hr26.x, g[r8.z], 1
ldg.u16 hr26.y, g[r10.y], 1
ldg.u16 hr26.w, g[r10.w], 1
(ss)mov.u32u32 r6.z, r53.y
mov.u32u32 r6.w, r54.x
ldg.u16 hr19.y, g[r15.z], 1
mov.u32u32 r8.z, r53.z
mov.u32u32 r8.w, r54.y
mov.u32u32 r10.y, r52.w
mov.u32u32 r10.z, r54.z
(rpt1)nop
ldg.u16 hr19.z, g[r6.z], 1
(rpt1)nop
ldg.u16 hr19.w, g[r8.z], 1
(rpt1)nop
ldg.u16 hr21.z, g[r10.y], 1
predt
shl.b r11.x, r12.w, 1
(nop2) shl.b r14.z, (last)r14.z, 1
add.u r11.x, (last)r11.x, -504
(nop2) shrg r13.z, 31, r13.z, (last)r14.z
(ss)(nop3) add.u r15.z, c0.z, (last)r11.x
(nop3) cmps.u.lt hr13.x, r15.z, c0.z
cov.u16s32 r6.w, (last)hr13.x
(rpt2)nop
(nop3) sad.s32 r15.w, c0.w, r13.z, (last)r6.w
(rpt2)nop
ldg.u16 hr17.x, g[r15.z], 4
predf
(rpt4)nop
(ss)mov.u16u16 hr17.x, 0
mov.u16u16 hr17.w, 0
mov.u16u16 hr17.z, 0
mov.u16u16 hr17.y, 0
prede
(rpt6)nop
add.u r52.y, r52.y, 1
cmps.s.ge up0.y, r52.y, 24
shl.b r10.y, r12.w, 1
shl.b r15.x, (last)r15.x, 1
shl.b r14.w, (last)r14.w, 1
shl.b r15.y, (last)r15.y, 1
sad.s32 r14.z, c4.w, r10.y, c0.z
shrg r14.x, 31, r14.x, (last)r15.x
sad.s32 r15.x, c5.z, r10.y, c0.z
ashr.b r10.z, r12.w, 31
cmps.u.lt hr20.y, r14.z, c0.z
(ss)sad.s32 r15.z, c5.w, r10.y, c0.z
shrg r13.w, 31, r13.w, (last)r14.w
cmps.u.lt hr20.w, r15.x, c0.z
cov.u16s32 r11.x, (last)hr20.y
sad.s32 r6.z, r12.w, r12.w, c0.z
shrg r14.y, 31, r14.y, (last)r15.y
shl.b r10.z, (last)r10.z, 1
cmps.u.lt hr20.z, r15.z, c0.z
cov.u16s32 r13.z, (last)hr20.w
sad.s32 r14.w, c0.w, r13.w, (last)r11.x
shrg r10.z, 31, r12.w, (last)r10.z
cov.u16s32 r12.w, (last)hr20.z
sad.s32 r15.y, c0.w, r14.y, (last)r13.z
(nop1) cmps.u.lt hr28.z, r6.z, c0.z
sad.s32 r15.w, c0.w, r14.x, (last)r12.w
ldg.u16 hr27.z, g[r14.z], 4
nop
(ss)cov.u16s32 r14.z, (last)hr28.z
(rpt2)nop
sad.s32 r6.w, c0.w, r10.z, (last)r14.z
ldg.u16 hr31.x, g[r15.z], 4
ldg.u16 hr20.y, g[r15.x], 4
(rpt5)nop
ldg.u16 hr28.z, g[r6.z], 4
(sy)mul.f r14.w, hr25.x, hr1.x
mul.f r15.x, hr26.z, hr21.z
mul.f r15.y, hr18.z, hr16.y
mul.f r6.z, hr2.x, hr15.w
mul.f r11.x, hr22.w, hr14.w
add.f r15.x, (last)r15.x, (last)r14.w
mul.f r12.w, hr26.z, hr16.z
mul.f r13.z, hr18.z, hr23.x
mul.f r14.w, hr2.x, hr23.y
add.f r15.x, (last)r15.x, (last)r15.y
mul.f r15.y, hr20.x, hr24.w
(nop1) mul.f r6.w, hr29.y, hr18.x
add.f r15.x, (last)r15.x, (last)r6.z
(nop3) mul.f r6.z, hr28.z, hr24.x
add.f r15.x, (last)r15.x, (last)r6.z
(nop3) mul.f r6.z, hr28.w, hr14.z
add.f r15.x, (last)r15.x, (last)r6.z
(nop2) mul.f r6.z, hr27.z, hr19.z
add.f r15.x, (last)r15.x, (last)r11.x
(nop2) mul.f r11.x, hr25.x, hr16.w
add.f r15.x, (last)r15.x, (last)r6.z
add.f r12.w, (last)r12.w, (last)r11.x
mul.f r6.z, hr27.w, hr19.w
(nop1) mul.f r11.x, hr22.w, hr15.x
add.f r12.w, (last)r12.w, (last)r13.z
add.f r15.x, (last)r15.x, (last)r6.z
mul.f r6.z, hr28.w, hr23.w
mul.f r13.z, hr17.w, hr2.z
add.f r12.w, (last)r12.w, (last)r14.w
add.f r6.y, (last)r6.y, (last)r15.x
mul.f r15.x, hr28.z, hr23.z
(nop1) mul.f r14.w, hr27.z, hr15.y
cov.f32f16 hr21.y, r6.y
add.f r12.w, (last)r12.w, (last)r15.x
(nop2) mul.f r15.x, hr28.y, hr19.x
add.f r12.w, (last)r12.w, (last)r6.z
(nop2) mul.f r6.z, hr27.w, hr15.z
add.f r12.w, (last)r12.w, (last)r11.x
(nop2) mul.f r11.x, hr16.x, hr24.y
add.f r12.w, (last)r12.w, (last)r14.w
(nop2) mul.f r14.w, hr17.x, hr24.z
add.f r12.w, (last)r12.w, (last)r6.z
add.f r11.x, (last)r11.x, (last)r14.w
mul.f r14.w, hr20.y, hr18.y
mul.f r6.z, hr20.z, hr18.w
add.f r6.x, (last)r6.x, (last)r12.w
add.f r11.x, (last)r11.x, (last)r15.y
(nop1) mul.f r12.w, hr20.z, hr23.y
cov.f32f16 hr21.w, r6.x
add.f r11.x, (last)r11.x, (last)r6.w
(nop2) mul.f r6.w, hr20.x, hr2.z
add.f r11.x, (last)r11.x, (last)r14.w
(nop2) mul.f r14.w, hr31.y, hr14.y
add.f r11.x, (last)r11.x, (last)r6.z
(nop2) mul.f r6.z, hr31.x, hr14.x
add.f r11.x, (last)r11.x, (last)r15.x
(nop2) mul.f r15.x, hr29.y, hr2.w
add.f r11.x, (last)r11.x, (last)r6.z
(nop2) mul.f r6.z, hr16.x, hr1.y
add.f r11.x, (last)r11.x, (last)r14.w
(nop2) mul.f r14.w, hr17.x, hr2.y
add.f r5.w, (last)r5.w, (last)r11.x
add.f r6.z, (last)r6.z, (last)r14.w
mul.f r14.w, hr20.y, hr25.y
mul.f r11.x, hr20.z, hr15.w
cov.f32f16 hr30.z, r5.w
add.f r6.z, (last)r6.z, (last)r6.w
(nop2) mul.f r6.w, hr17.w, hr16.y
add.f r6.z, (last)r6.z, (last)r15.x
(nop2) mul.f r15.x, hr28.y, hr26.y
add.f r6.z, (last)r6.z, (last)r14.w
(nop3) mul.f r14.w, hr20.z, hr26.x
add.f r6.z, (last)r6.z, (last)r14.w
(nop2) mul.f r14.w, hr31.x, hr26.w
add.f r6.z, (last)r6.z, (last)r15.x
(nop2) mul.f r15.x, hr17.z, hr1.x
add.f r6.z, (last)r6.z, (last)r14.w
(nop3) mul.f r14.w, hr31.y, hr19.y
add.f r6.z, (last)r6.z, (last)r14.w
(nop2) mul.f r14.w, hr20.z, hr18.x
add.f r5.z, (last)r5.z, (last)r6.z
(nop2) mul.f r6.z, hr17.y, hr21.z
cov.f32f16 hr30.w, r5.z
add.f r6.z, (last)r6.z, (last)r15.x
(nop2) mul.f r15.x, hr20.w, hr24.x
add.f r6.z, (last)r6.z, (last)r6.w
(nop2) mul.f r6.w, hr31.y, hr14.w
add.f r6.z, (last)r6.z, (last)r11.x
(nop2) mul.f r11.x, hr17.w, hr23.x
add.f r6.z, (last)r6.z, (last)r15.x
(nop3) mul.f r15.x, hr21.x, hr14.z
add.f r6.z, (last)r6.z, (last)r15.x
(nop2) mul.f r15.x, hr31.z, hr19.z
add.f r6.z, (last)r6.z, (last)r6.w
(nop2) mul.f r6.w, hr17.y, hr16.z
add.f r6.z, (last)r6.z, (last)r15.x
(nop3) mul.f r15.x, hr31.w, hr19.w
add.f r6.z, (last)r6.z, (last)r15.x
(nop2) mul.f r15.x, hr17.z, hr16.w
add.f r5.y, (last)r5.y, (last)r6.z
add.f r6.w, (last)r6.w, (last)r15.x
(nop1) mul.f r15.x, hr20.w, hr23.z
cov.f32f16 hr13.x, r5.y
add.f r6.w, (last)r6.w, (last)r11.x
(nop2) mul.f r11.x, hr31.y, hr15.x
add.f r6.w, (last)r6.w, (last)r12.w
(nop2) mul.f r12.w, (last)hr17.w, hr24.w
add.f r6.w, (last)r6.w, (last)r15.x
(nop3) mul.f r15.x, hr21.x, hr23.w
add.f r6.w, (last)r6.w, (last)r15.x
(nop2) mul.f r15.x, hr31.z, hr15.y
add.f r6.w, (last)r6.w, (last)r11.x
(nop2) mul.f r11.x, hr17.y, hr1.y
add.f r6.w, (last)r6.w, (last)r15.x
(nop3) mul.f r15.x, hr31.w, hr15.z
add.f r6.w, (last)r6.w, (last)r15.x
(nop2) mul.f r15.x, hr17.z, hr24.z
add.f r5.x, (last)r5.x, (last)r6.w
(nop2) mul.f r6.w, (last)hr17.y, hr24.y
cov.f32f16 hr13.y, r5.x
add.f r6.w, (last)r6.w, (last)r15.x
mul.f r15.x, (last)hr17.z, hr2.y
(nop1) mul.f r8.w, hr21.x, hr18.w
add.f r6.w, (last)r6.w, (last)r12.w
add.f r11.x, (last)r11.x, (last)r15.x
mul.f r15.x, hr20.w, hr18.y
mul.f r12.w, (last)hr21.x, hr26.x
add.f r6.w, (last)r6.w, (last)r14.w
add.f r11.x, (last)r11.x, (last)r13.z
mul.f r13.z, hr31.y, hr19.x
mul.f r14.w, hr31.z, hr14.x
add.f r6.w, (last)r6.w, (last)r15.x
(nop2) mul.f r15.x, (last)hr20.w, hr25.y
add.f r6.w, (last)r6.w, (last)r8.w
(nop2) mul.f r8.w, (last)hr31.z, hr26.w
add.f r6.w, (last)r6.w, (last)r13.z
(nop2) mul.f r13.z, hr31.w, hr14.y
add.f r6.w, (last)r6.w, (last)r14.w
(nop2) mul.f r14.w, (last)hr31.w, hr19.y
(nop3) add.f r6.w, (last)r6.w, (last)r13.z
add.f r4.w, (last)r4.w, (last)r6.w
(nop2) mul.f r6.w, hr20.z, hr2.w
cov.f32f16 hr31.z, r4.w
add.f r11.x, (last)r11.x, (last)r6.w
(nop2) mul.f r6.w, hr18.z, hr24.w
add.f r11.x, (last)r11.x, (last)r15.x
(nop2) mul.f r15.x, hr26.z, hr24.y
add.f r11.x, (last)r11.x, (last)r12.w
(nop3) mul.f r12.w, hr31.y, hr26.y
add.f r11.x, (last)r11.x, (last)r12.w
(nop2) mul.f r12.w, hr28.z, hr25.y
add.f r11.x, (last)r11.x, (last)r8.w
(nop2) mul.f r8.w, hr2.x, hr18.x
add.f r11.x, (last)r11.x, (last)r14.w
(nop2) mul.f r14.w, hr25.x, hr24.z
add.f r4.z, (last)r4.z, (last)r11.x
add.f r15.x, (last)r15.x, (last)r14.w
mul.f r11.x, (last)hr28.z, hr18.y
mul.f r14.w, hr28.w, hr18.w
cov.f32f16 hr27.x, r4.z
add.f r15.x, (last)r15.x, (last)r6.w
(nop2) mul.f r6.w, hr22.w, hr19.x
add.f r15.x, (last)r15.x, (last)r8.w
(nop2) mul.f r8.w, hr27.z, hr14.x
add.f r15.x, (last)r15.x, (last)r11.x
(nop2) mul.f r11.x, (last)hr27.z, hr26.w
add.f r15.x, (last)r15.x, (last)r14.w
(nop2) mul.f r14.w, hr27.w, hr14.y
add.f r15.x, (last)r15.x, (last)r6.w
(nop2) mul.f r6.w, (last)hr25.x, hr2.y
add.f r15.x, (last)r15.x, (last)r8.w
(nop2) mul.f r8.w, (last)hr26.z, hr1.y
add.f r15.x, (last)r15.x, (last)r14.w
add.f r8.w, (last)r8.w, (last)r6.w
mul.f r14.w, hr18.z, hr2.z
mul.f r6.w, (last)hr2.x, hr2.w
(nop1) add.f r4.y, (last)r4.y, (last)r15.x
add.f r8.w, (last)r8.w, (last)r14.w
mul.f r14.w, (last)hr22.w, hr26.y
cov.f32f16 hr30.x, r4.y
nop
add.f r8.w, (last)r8.w, (last)r6.w
(nop2) mul.f r6.w, hr27.w, hr19.y
add.f r8.w, (last)r8.w, (last)r12.w
(nop3) mul.f r12.w, hr28.w, hr26.x
add.f r8.w, (last)r8.w, (last)r12.w
(nop2) mul.f r12.w, hr16.x, hr16.y
add.f r8.w, (last)r8.w, (last)r14.w
(nop2) mul.f r14.w, hr28.w, hr15.w
add.f r8.w, (last)r8.w, (last)r11.x
(nop2) mul.f r11.x, hr18.z, hr21.z
add.f r8.w, (last)r8.w, (last)r6.w
(nop2) mul.f r6.w, hr29.x, hr24.x
add.f r4.x, (last)r4.x, (last)r8.w
(nop2) mul.f r8.w, hr22.z, hr1.x
cov.f32f16 hr17.y, r4.x
(nop3) add.f r11.x, (last)r11.x, (last)r8.w
add.f r11.x, (last)r11.x, (last)r12.w
(nop2) mul.f r12.w, hr18.z, hr16.z
add.f r11.x, (last)r11.x, (last)r14.w
(nop2) mul.f r14.w, hr16.x, hr23.x
add.f r11.x, (last)r11.x, (last)r6.w
(nop3) mul.f r6.w, hr29.y, hr14.z
add.f r11.x, (last)r11.x, (last)r6.w
(nop3) mul.f r6.w, hr27.w, hr14.w
add.f r11.x, (last)r11.x, (last)r6.w
(nop3) mul.f r6.w, hr28.x, hr19.z
add.f r11.x, (last)r11.x, (last)r6.w
(nop3) mul.f r6.w, hr28.y, hr19.w
add.f r11.x, (last)r11.x, (last)r6.w
(nop2) mul.f r6.w, hr28.w, hr23.y
add.f r3.w, (last)r3.w, (last)r11.x
(nop2) mul.f r11.x, hr22.z, hr16.w
cov.f32f16 hr17.z, r3.w
add.f r12.w, (last)r12.w, (last)r11.x
(nop2) mul.f r11.x, hr29.x, hr23.z
add.f r12.w, (last)r12.w, (last)r14.w
(nop2) mul.f r14.w, hr27.w, hr15.x
add.f r12.w, (last)r12.w, (last)r6.w
(nop2) mul.f r6.w, hr29.y, hr23.w
add.f r12.w, (last)r12.w, (last)r11.x
(nop2) mul.f r11.x, hr22.z, (last)hr24.z
add.f r12.w, (last)r12.w, (last)r6.w
(nop2) mul.f r6.w, hr28.x, hr15.y
add.f r12.w, (last)r12.w, (last)r14.w
mul.f r14.w, hr16.x, (last)hr24.w
(nop1) mul.f r12.y, hr29.y, (last)hr18.w
add.f r12.w, (last)r12.w, (last)r6.w
(nop3) mul.f r6.w, hr28.y, hr15.z
add.f r12.w, (last)r12.w, (last)r6.w
mul.f r6.w, hr28.w, (last)hr18.x
(nop1) mul.f r9.x, hr29.x, (last)hr18.y
add.f r3.z, (last)r3.z, (last)r12.w
mul.f r12.w, hr18.z, (last)hr24.y
(nop1) mul.f r9.y, (last)hr18.z, (last)hr1.y
cov.f32f16 hr17.w, r3.z
add.f r12.w, (last)r12.w, (last)r11.x
mul.f r11.x, (last)hr29.x, (last)hr25.y
mul.f r12.z, hr27.w, (last)hr19.x
mul.f r13.w, (last)hr27.w, (last)hr26.y
add.f r12.w, (last)r12.w, (last)r14.w
mul.f r14.w, (last)hr28.w, (last)hr2.w
mul.f r13.x, hr29.y, (last)hr26.x
mul.f r14.y, hr28.x, (last)hr26.w
add.f r12.w, (last)r12.w, (last)r6.w
mul.f r6.w, hr28.y, (last)hr14.y
(nop1) mul.f r13.y, hr29.y, (last)hr15.w
add.f r12.w, (last)r12.w, (last)r9.x
(nop2) mul.f r9.x, (last)hr28.x, (last)hr14.x
add.f r12.w, (last)r12.w, (last)r12.y
(nop2) mul.f r12.y, hr28.y, (last)hr19.y
(nop3) add.f r12.w, (last)r12.w, (last)r12.z
add.f r12.w, (last)r12.w, (last)r9.x
mul.f r9.x, (last)hr22.z, (last)hr2.y
mul.f r11.y, hr16.x, (last)hr2.z
mul.f r1.y, hr28.y, (last)hr14.w
add.f r12.w, (last)r12.w, (last)r6.w
add.f r9.y, (last)r9.y, (last)r9.x
mul.f r9.x, hr31.y, (last)hr19.w
mul.f r1.x, hr20.z, (last)hr23.w
add.f r3.y, (last)r3.y, (last)r12.w
add.f r9.y, (last)r9.y, (last)r11.y
mul.f r12.w, hr16.x, (last)hr21.z
mul.f r6.w, (last)hr28.y, (last)hr15.x
mul.f r7.z, hr31.x, (last)hr15.y
add.f r9.y, (last)r9.y, (last)r14.w
mul.f r14.w, hr17.x, (last)hr1.x
mul.f r0.z, (last)hr20.z, (last)hr14.z
mul.f r7.y, (last)hr31.x, (last)hr19.z
add.f r9.y, (last)r9.y, (last)r11.x
add.f r12.w, (last)r12.w, (last)r14.w
mul.f r9.w, (last)hr16.x, (last)hr16.z
mul.f r14.w, (last)hr17.x, (last)hr16.w
add.f r9.y, (last)r9.y, (last)r13.x
mul.f r13.x, hr20.x, (last)hr16.y
cov.f32f16 hr14.x, r3.y
add.f r9.w, (last)r9.w, (last)r14.w
add.f r9.y, (last)r9.y, (last)r13.w
mul.f r13.w, (last)hr29.y, (last)hr23.y
mul.f r14.z, hr20.y, (last)hr23.z
add.f r12.w, (last)r12.w, (last)r13.x
add.f r9.y, (last)r9.y, (last)r14.y
mul.f r14.y, (last)hr20.y, (last)hr24.x
mul.f r10.x, (last)hr20.x, (last)hr23.x
add.f r12.w, (last)r12.w, (last)r13.y
(nop1) add.f r9.y, (last)r9.y, (last)r12.y
add.f r9.w, (last)r9.w, (last)r10.x
add.f r12.w, (last)r12.w, (last)r14.y
add.f r3.x, (last)r3.x, (last)r9.y
mul.f r9.y, (last)hr31.y, (last)hr15.z
add.f r9.w, (last)r9.w, (last)r13.w
add.f r12.w, (last)r12.w, (last)r0.z
cov.f32f16 hr25.x, r3.x
nop
add.f r9.w, (last)r9.w, (last)r14.z
(nop2) add.f r12.w, (last)r12.w, (last)r1.y
add.f r9.w, (last)r9.w, (last)r1.x
(nop2) add.f r12.w, (last)r12.w, (last)r7.y
add.f r9.w, (last)r9.w, (last)r6.w
(nop2) add.f r12.w, (last)r12.w, (last)r9.x
add.f r9.w, (last)r9.w, (last)r7.z
(nop2) add.f r2.w, (last)r2.w, (last)r12.w
add.f r9.w, (last)r9.w, (last)r9.y
cov.f32f16 hr19.x, r2.w
(rpt1)nop
(nop3) add.f r2.z, (last)r2.z, (last)r9.w
cov.f32f16 hr20.z, r2.z
br p0.y, #l1073
cmps.s.lt p0.x, 0, r0.x
cmps.s.lt p0.y, 0, r0.y
jump #l80
l1073:
(jp)add.u r52.x, c1.z, r52.x
cmps.u.lt hr50.x, r52.x, c1.z
cov.u16s32 r49.y, hr50.x
add.u r49.y, r49.y, c1.w
shl.b r48.y, r48.y, 11
shl.b r48.z, r48.z, 15
(ss)mov.u32u32 r11.x, r52.x
mov.u32u32 r11.y, r49.y
(nop3) add.u r1.z, r48.y, (last)r1.z
(nop1) sad.s32 r2.y, (last)r2.y, r1.z, r48.z
ldg.u16 hr23.x, g[r11.x], 4
(rpt1)nop
add.u r14.y, r2.y, c2.y
add.u r14.z, r2.y, c6.y
add.u r14.w, r2.y, c11.x
ashr.b r1.z, r2.y, 31
sad.s32 r1.w, r2.y, r2.y, c0.x
(nop1) ashr.b r15.z, r14.z, 31
shl.b r1.z, (last)r1.z, 1
cmps.u.lt hr7.x, r1.w, c0.x
(nop1) shl.b r15.z, (last)r15.z, 1
shrg r1.z, 31, r2.y, (last)r1.z
cov.u16s32 r3.w, (last)hr7.x
(nop2) shrg r14.z, 31, r14.z, (last)r15.z
sad.s32 r2.x, c0.y, r1.z, (last)r3.w
(sy)add.f hr24.x, (last)hr30.x, hr23.z
shl.b r15.x, (last)r2.y, 1
add.f hr24.z, (last)hr30.z, hr23.z
add.f hr24.w, (last)hr31.z, hr23.z
add.f hr25.w, (last)hr30.w, hr23.w
sad.s32 r0.x, c2.x, r15.x, c0.x
sad.s32 r1.x, c6.x, r15.x, c0.x
sad.s32 r0.z, c2.y, r15.x, c0.x
ashr.b r15.w, r14.w, 31
ashr.b r15.y, r14.y, 31
cmps.u.lt hr4.z, r0.x, c0.x
cmps.u.lt hr4.w, r1.x, c0.x
cmps.u.lt hr5.x, r0.z, c0.x
shl.b r15.w, (last)r15.w, 1
shl.b r15.y, (last)r15.y, 1
(rpt2)cov.u16s32 r2.w, (last)(r)hr4.z
shrg r14.w, 31, r14.w, (last)r15.w
shrg r14.y, 31, r14.y, (last)r15.y
add.f hr24.y, (last)hr14.x, (last)hr23.z
add.f hr26.x, (last)hr27.x, hr23.w
add.f hr25.y, (last)hr17.y, hr23.w
add.f hr25.z, (last)hr25.x, (last)hr23.w
add.f hr26.y, (last)hr21.w, hr23.y
add.f hr26.z, (last)hr17.w, hr23.y
add.f hr26.w, (last)hr20.z, hr23.y
add.f hr27.y, (last)hr21.y, hr23.x
add.f hr28.x, (last)hr13.x, hr23.x
add.f hr27.z, (last)hr17.z, hr23.x
add.f hr27.w, (last)hr19.x, (last)hr23.x
sad.s32 r1.y, c0.y, r14.z, (last)r3.x
sad.s32 r0.w, c0.y, r14.w, (last)r3.y
sad.s32 r0.y, c0.y, r14.y, (last)r2.w
(nop3) add.f hr27.x, (last)hr13.y, (last)hr23.y
(rpt1)nop
stg.u16 g[r0.x], hr24.x, 4
stg.u16 g[r1.x], hr25.y, 4
nop
stg.u16 g[r0.z], hr26.y, 4
stg.u16 g[r1.w], hr27.y, 4
end
nop
nop
nop
nop
nop
nop
nop
nop
nop
nop
nop
nop
nop
nop
