set frame_importance_file [ lindex $argv 0 ]
set pdb [ lindex $argv 1 ]
set xtc_file [ lindex $argv 2 ]
set videoname [ lindex $argv 3 ]
set smoothsize 5
#set outfile [ lindex $argv 1 ]
mol new $pdb type pdb waitfor all
mol addfile $xtc_file type {xtc} first 0 last -1 step 1 waitfor all

proc color_scheme {} {
  set color_start [colorinfo num]
  display update off
  for {set i 0} {$i < 1024} {incr i} {
    # from WHITE to BLUE
    set r [expr 1-$i/1024.] ;  set g [expr 1-$i/1024.] ; set b 1
    color change rgb [expr $i + $color_start ] $r $g $b }
  display update on 
}

color_scheme

display update on
axes location off
display projection orthographic
display resetview
color Display Background white
display shadows on
display ambientocclusion on

# Load protein
mol delrep 0 top
mol selection {protein}
mol representation NewCartoon
mol color User
mol addrep top
mol modmaterial 0 top AOChalky
mol colupdate 0 0 0
mol selupdate 0 0 1
mol scaleminmax 0 0 0.300000 0.700000

#show ligand
mol addrep 0
mol modselect 1 0 not protein and noh
mol modstyle 1 0 Licorice 0.900000 12.000000 12.000000
mol modcolor 1 0 ColorID 2
mol modmaterial 1 0 AOChalky

#Scale and roate
rotate x by -90
rotate y by -50
#rotate z by 180
scale by 2.2
material change ambient AOChalky 0.1
material change outline AOChalky 1.4
display depthcue off
display rendermode GLSL
display update


#################Animation
####For Movie rendering##########33
#Smooth with 5 frames per window to remove some thermal vibrations
mol smoothrep 0 0 $smoothsize
mol smoothrep 0 1 $smoothsize
mol scaleminmax 0 0 0.300000 0.700000
#menu vmdmovie on
#############LOAD data into user field
##Below from https://www.ks.uiuc.edu/Research/vmd/mailing_list/vmd-l/5001.html
#The ke file is just a long list of ke 
#for each atom, and each frame is cancatinated to the file. This 
#script then applies each ke to each atom, frame after frame
#script then applies each ke to each atom, frame after frame
set protein_sel [atomselect [molinfo top] {protein}]
set numatoms [$protein_sel num]
set numframes [molinfo top get numframes] 
puts "Setting importance on $numframes frames and $numatoms atoms"
# Skip first frame since it is the topology usually
set frame_offset 1

## load all importances into memory
set frame_importances [open $frame_importance_file r] 
set all_importance {}
for {set i $frame_offset} {$i<($numframes)} {incr i} {   
  if {$i%100==0} {
    puts "Loading importance data for frame $i/$numframes..." 
  }
  # list with all importance for this frame
  set fis {} 
  for {set j 0} {$j<($numatoms)} {incr j} { 
    set fi [gets $frame_importances] 
    if {[string first "#" $fi] != -1} {
      #puts "Found next frame, leaving loop"
      if {$j==0} {
        #We are reading the header of this frame, just read the next line
        set fi [gets $frame_importances] 
       } else {
        puts "#Number of atoms in toplogy does not match number of atoms in the file and we reached the next frame -> will go to next frame"
        break 
       }
    }
    lappend fis $fi
  }
  lappend all_importance $fis
} 

## Set the color of every atom at every frame.
for {set i $frame_offset} {$i<($numframes)} {incr i} {   
  animate goto $i 
  [atomselect top "all"] set user 0
  for {set j 0} {$j<($numatoms)} {incr j} { 
    set fi 0
    #Average over a number of frames
    set nvals 0
    for {set k 0} {$k <($smoothsize)} {incr k} {
      set val [lindex $all_importance [expr $i + $k - $frame_offset] $j]
      if {$val < -999} {
        break
      }
      set fi [expr $fi + $val]
      set nvals [expr $nvals + 1]
    }
    set fi [expr $fi/($nvals + 0.000001)]
    if {$fi<0.1} {continue}
    #puts "Settings frame importance "
    set atomsel [atomselect top "index $j" frame $i] 
    $atomsel set user $fi
    $atomsel delete 
  }
  if {$i%100==0} {
    puts "Setting 'User' field on atoms in frame $i/$numframes..." 
    display update
  } 
  #if {$i > 10} {break}
} 

display update
color_scheme
menu tkcon on
puts "run color_scheme if colors looks weird"

proc make_trajectory_movie_files {molname frame_offset} {
  set num [molinfo top get numframes]
  puts "rendering trajectory with $num frames"
  #set lst [split $pdb /]
  #set molname [lindex $lst 1]
  # loop through the frames
  set counter 0
  #Iterating backwards below
  for {set i $num} {$i >= $frame_offset} {incr i -1} {
    # go to the given frame
    animate goto $i
    # force display update
    display update 
    #take picture of frame
    set outfile [format "trajectories/videooutput/$molname/frame.%05d.ppm" $counter]
    puts "Rendering frame $i to file $outfile"
    render TachyonLOptiXInternal $outfile 
    incr counter
  }
  puts "To render run the following command in your terminal: "
  #puts "to convert to AVI format, run: 'ffmpeg -i input.mp4 -c:v libx264 -c:a libmp3lame -b:a 384K output.avi'"
  puts "ffmpeg -framerate 60 -pattern_type glob -i \"trajectories/videooutput/$molname/frame.*.ppm\" -c:v libx264 -preset slow -crf 18 -r 60 trajectories/videooutput/$molname-mlp.mp4"
}
puts "run make_trajectory_movie_files \$videoname \$frame_offset to generate movie"

#render Tachyon [format $pdb.dat] 
#/usr/local/lib/vmd/tachyon_LINUXAMD64 $pdb.dat -format TARGA -res 630 1000 -o $outfile.tga
#exit
