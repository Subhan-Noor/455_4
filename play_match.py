import subprocess
import sys
import os
import select

p1_name = sys.argv[1]
p2_name = sys.argv[2]

if not os.path.isfile(p1_name):
    print("Could not find file", p1_name)
if not os.path.isfile(p2_name):
    print("Could not find file", p2_name)

p1_proc = subprocess.Popen(["python3", p1_name], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
p2_proc = subprocess.Popen(["python3", p2_name], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

xsize = 7
ysize = 7
handicap = 5.5
score_cutoff = 0
timelimit = 1

def send_command(process, name, command):
    print(name, ":", command)
    process.stdin.write(command+"\n")
    process.stdin.flush()
    output = ""
    line = process.stdout.readline()
    try:
        while line[0] != "=":
            #print(line)
            if len(line.strip()) > 0:
                output += line
            line = process.stdout.readline()
            readable, _, _ = select.select([process.stderr], [], [], 0)
            if readable:
                sys.stderr.write(process.stderr.read())
                sys.stderr.flush()
    except:
        sys.exit()
    print(output)
    return output

def play_game(p1, p1_name, p2, p2_name):
    print("P1:", p1_name)
    print("P2:", p2_name)
    params = " ".join([str(p) for p in [xsize, ysize, handicap, score_cutoff]])
    send_command(p1, p1_name, "init_game "+params)
    send_command(p2, p2_name, "init_game "+params)
    send_command(p1, p1_name, "timelimit "+str(timelimit))
    send_command(p2, p2_name, "timelimit "+str(timelimit))
    if score_cutoff == 0:
        real_score_cutoff = float('inf')

    player = p1
    p_name = p1_name
    opp = p2
    o_name = p2_name
    won = False
    for _ in range(xsize*ysize):
        move = send_command(player, p_name, "genmove").strip()
        send_command(opp, o_name, "play " + move)
        send_command(player, p_name, "show")
        score = send_command(player, p_name, "score")
        p1_score, p2_score = [float(x) for x in score.split(" ")]
        if p1_score > real_score_cutoff:
            print(p1_name, "wins by score cutoff.")
            return p1
        if p2_score > real_score_cutoff:
            print(p2_name, "wins by score cutoff.")
            return p2
        tmp = player
        tmp_name = p_name
        player = opp
        p_name = o_name
        opp = tmp
        o_name = tmp_name
    if p1_score > p2_score:
        print(p1_name, "wins by", p1_score-p2_score, "points.")
    else:
        print(p2_name, "wins by", p2_score-p1_score, "points.")
    return p1_score-p2_score

p1_score = play_game(p1_proc, p1_name, p2_proc, p2_name)
p1_score += -play_game(p2_proc, p2_name, p1_proc, p1_name)

print("Overall winner:")
if p1_score > 0:
    print(p1_name, "by a", p1_score, "point difference.")
else:
    print(p2_name, "by a", -p1_score, "point difference.")