set pdb [ lindex $argv 0 ]
set outfile [ lindex $argv 1 ]
mol new $pdb type pdb waitfor all

proc color_scheme {} {
  set color_start [colorinfo num]
  display update off
  for {set i 0} {$i < 1024} {incr i} {
    # from WHITE to BLUE
    set r [expr 1-$i/1024.] ;  set g [expr 1-$i/1024.] ; set b 1
    color change rgb [expr $i + $color_start ] $r $g $b }
  display update on }

color_scheme

display update on
axes location off
display projection orthographic
display resetview
color Display Background white
display shadows on
display ambientocclusion on
mol delrep 0 top
mol selection {protein}
mol representation NewCartoon
mol color Beta
mol addrep top
mol modmaterial 0 top AOChalky
mol scaleminmax top 0 0 1
rotate x by -90
#rotate y by -45
rotate z by 180
# [atomselect top "all"] moveby {0 4 0}
scale by 1.65
material change ambient AOChalky 0.1
material change outline AOChalky 1.4
display depthcue off
display rendermode GLSL
display update

render Tachyon [format $pdb.dat] 
/usr/local/lib/vmd/tachyon_LINUXAMD64 $pdb.dat -format TARGA -res 630 1000 -o $outfile.tga
exit
