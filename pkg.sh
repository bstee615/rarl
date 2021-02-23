#!/bin/bash
name=$1
ssh benjis@prontodtn.las.iastate.edu "cd /work/LAS/weile-lab/benjis/weile-lab/rarl/ && tar -zcf $1.tar.gz models/ logs/"
scp "benjis@prontodtn.las.iastate.edu:/work/LAS/weile-lab/benjis/weile-lab/rarl/$1.tar.gz" .
tar --transform "s@logs/@logs/$1/@" -zxf $1.tar.gz logs/

