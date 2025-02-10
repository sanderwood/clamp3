#!/usr/bin/env python
# coding=latin-1
'''
Copyright (C) 2012-2018: Willem G. Vree
Contributions: Nils Liberg, Nicolas Froment, Norman Schmidt, Reinier Maliepaard, Martin Tarenskeen,
               Paul Villiger, Alexander Scheutzow, Herbert Schneider, David Randolph, Michael Strasser

This program is free software; you can redistribute it and/or modify it under the terms of the
Lesser GNU General Public License as published by the Free Software Foundation;

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the Lesser GNU General Public License for more details. <http://www.gnu.org/licenses/lgpl.html>.
'''

from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
try:    import xml.etree.cElementTree as E
except: import xml.etree.ElementTree as E
import types, sys, os, re, datetime

VERSION = 245

python3 = sys.version_info[0] > 2
lmap = lambda f, xs: list (map (f, xs))   # eager map for python 3
if python3:
    int_type = int
    list_type = list
    str_type = str
    uni_type = str
    stdin = sys.stdin.buffer if sys.stdin else None  # read binary if stdin available!
else:
    int_type = types.IntType
    list_type = types.ListType
    str_type = types.StringTypes
    uni_type = types.UnicodeType
    stdin = sys.stdin

info_list = []  # diagnostic messages
def info (s, warn=1):
    x = (warn and '-- ' or '') + s
    info_list.append (x + '\n')         # collect messages
    if __name__ == '__main__':          # only write to stdout when called as main progeam
        try: sys.stderr.write (x + '\n')
        except: sys.stderr.write (repr (x) + '\n')

def getInfo (): # get string of diagnostic messages, then clear messages
    global info_list
    xs = ''.join (info_list)
    info_list = []
    return xs

def abc_grammar ():     # header, voice and lyrics grammar for ABC
    #-----------------------------------------------------------------
    # expressions that catch and skip some syntax errors (see corresponding parse expressions)
    #-----------------------------------------------------------------
    b1 = Word (u"-,'<>\u2019#", exact=1)    # catch misplaced chars in chords
    b2 = Regex ('[^H-Wh-w~=]*')             # same in user defined symbol definition
    b3 = Regex ('[^=]*')                    # same, second part

    #-----------------------------------------------------------------
    # ABC header (field_str elements are matched later with reg. epr's)
    #-----------------------------------------------------------------

    number = Word (nums).setParseAction (lambda t: int (t[0]))
    field_str = Regex (r'[^]]*')  # match anything until end of field
    field_str.setParseAction (lambda t: t[0].strip ())  # and strip spacing

    userdef_symbol  = Word (srange ('[H-Wh-w~]'), exact=1)
    fieldId = oneOf ('K L M Q P I T C O A Z N G H R B D F S E r Y') # info fields
    X_field = Literal ('X') + Suppress (':') + field_str
    U_field = Literal ('U') + Suppress (':') + b2 + Optional (userdef_symbol, 'H') + b3 + Suppress ('=') + field_str
    V_field = Literal ('V') + Suppress (':') + Word (alphanums + '_') + field_str
    inf_fld = fieldId + Suppress (':') + field_str
    ifield = Suppress ('[') + (X_field | U_field | V_field | inf_fld) + Suppress (']')
    abc_header = OneOrMore (ifield) + StringEnd ()

    #---------------------------------------------------------------------------------
    # I:score with recursive part groups and {* grand staff marker
    #---------------------------------------------------------------------------------

    voiceId = Suppress (Optional ('*')) + Word (alphanums + '_')
    voice_gr = Suppress ('(') + OneOrMore (voiceId | Suppress ('|')) + Suppress (')')
    simple_part = voiceId | voice_gr | Suppress ('|')
    grand_staff = oneOf ('{* {') + OneOrMore (simple_part) + Suppress ('}')
    part = Forward ()
    part_seq = OneOrMore (part | Suppress ('|'))
    brace_gr = Suppress ('{') + part_seq + Suppress ('}')
    bracket_gr = Suppress ('[') + part_seq + Suppress (']')
    part <<= MatchFirst (simple_part | grand_staff | brace_gr | bracket_gr | Suppress ('|'))
    abc_scoredef = Suppress (oneOf ('staves score')) + OneOrMore (part)

    #----------------------------------------
    # ABC lyric lines (white space sensitive)
    #----------------------------------------

    skip_note   = oneOf ('* -')
    extend_note = Literal ('_')
    measure_end = Literal ('|')
    syl_str     = CharsNotIn ('*-_| \t\n\\]')
    syl_chars   = Combine (OneOrMore (syl_str | Regex (r'\\.')))
    white       = Word (' \t')
    syllable    = syl_chars + Optional ('-')
    lyr_elem    = (syllable | skip_note | extend_note | measure_end) + Optional (white).suppress ()
    lyr_line    = Optional (white).suppress () + ZeroOrMore (lyr_elem)
    
    syllable.setParseAction (lambda t: pObj ('syl', t))
    skip_note.setParseAction (lambda t: pObj ('skip', t))
    extend_note.setParseAction (lambda t: pObj ('ext', t))
    measure_end.setParseAction (lambda t: pObj ('sbar', t))
    lyr_line_wsp = lyr_line.leaveWhitespace ()   # parse actions must be set before calling leaveWhitespace

    #---------------------------------------------------------------------------------
    # ABC voice (not white space sensitive, beams detected in note/rest parse actions)
    #---------------------------------------------------------------------------------

    inline_field =  Suppress ('[') + (inf_fld | U_field | V_field) + Suppress (']')
    lyr_fld = Suppress ('[') + Suppress ('w') + Suppress (':') + lyr_line_wsp + Suppress (']')  # lyric line
    lyr_blk = OneOrMore (lyr_fld)       # verses
    fld_or_lyr = inline_field | lyr_blk # inline field or block of lyric verses

    note_length = Optional (number, 1) + Group (ZeroOrMore ('/')) + Optional (number, 2)
    octaveHigh = OneOrMore ("'").setParseAction (lambda t: len(t))
    octaveLow = OneOrMore (',').setParseAction (lambda t: -len(t))
    octave  = octaveHigh | octaveLow

    basenote = oneOf ('C D E F G A B c d e f g a b y')  # includes spacer for parse efficiency
    accidental = oneOf ('^^ __ ^ _ =')
    rest_sym  = oneOf ('x X z Z')
    slur_beg = oneOf ("( (, (' .( .(, .('") + ~Word (nums)    # no tuplet_start
    slur_ends = OneOrMore (oneOf (') .)'))

    long_decoration = Combine (oneOf ('! +') + CharsNotIn ('!+ \n') + oneOf ('! +'))
    staccato        = Literal ('.') + ~Literal ('|')    # avoid dotted barline
    pizzicato       = Literal ('!+!')   # special case: plus sign is old style deco marker
    decoration      = slur_beg | staccato | userdef_symbol | long_decoration | pizzicato
    decorations     = OneOrMore (decoration)

    tie = oneOf ('.- -')
    rest = Optional (accidental) + rest_sym + note_length
    pitch = Optional (accidental) + basenote + Optional (octave, 0)
    note = pitch + note_length + Optional (tie) + Optional (slur_ends)
    dec_note = Optional (decorations) + pitch + note_length + Optional (tie) + Optional (slur_ends)
    chord_note = dec_note | rest | b1
    grace_notes = Forward ()
    chord = Suppress ('[') + OneOrMore (chord_note | grace_notes) + Suppress (']') + note_length + Optional (tie) + Optional (slur_ends)
    stem = note | chord | rest

    broken = Combine (OneOrMore ('<') | OneOrMore ('>'))

    tuplet_num   = Suppress ('(') + number
    tuplet_into  = Suppress (':') + Optional (number, 0)
    tuplet_notes = Suppress (':') + Optional (number, 0)
    tuplet_start = tuplet_num + Optional (tuplet_into + Optional (tuplet_notes))

    acciaccatura    = Literal ('/')
    grace_stem      = Optional (decorations) + stem
    grace_notes     <<= Group (Suppress ('{') + Optional (acciaccatura) + OneOrMore (grace_stem) + Suppress ('}'))

    text_expression  = Optional (oneOf ('^ _ < > @'), '^') + Optional (CharsNotIn ('"'), "")
    chord_accidental = oneOf ('# b =')
    triad            = oneOf ('ma Maj maj M mi min m aug dim o + -')
    seventh          = oneOf ('7 ma7 Maj7 M7 maj7 mi7 min7 m7 dim7 o7 -7 aug7 +7 m7b5 mi7b5')
    sixth            = oneOf ('6 ma6 M6 mi6 min6 m6')
    ninth            = oneOf ('9 ma9 M9 maj9 Maj9 mi9 min9 m9')
    elevn            = oneOf ('11 ma11 M11 maj11 Maj11 mi11 min11 m11')
    thirt            = oneOf ('13 ma13 M13 maj13 Maj13 mi13 min13 m13')
    suspended        = oneOf ('sus sus2 sus4')
    chord_degree     = Combine (Optional (chord_accidental) + oneOf ('2 4 5 6 7 9 11 13'))
    chord_kind       = Optional (seventh | sixth | ninth | elevn | thirt | triad) + Optional (suspended)
    chord_root       = oneOf ('C D E F G A B') + Optional (chord_accidental)
    chord_bass       = oneOf ('C D E F G A B') + Optional (chord_accidental) # needs a different parse action
    chordsym         = chord_root + chord_kind + ZeroOrMore (chord_degree) + Optional (Suppress ('/') + chord_bass)
    chord_sym        = chordsym + Optional (Literal ('(') + CharsNotIn (')') + Literal (')')).suppress ()
    chord_or_text    = Suppress ('"') + (chord_sym ^ text_expression) + Suppress ('"')

    volta_nums = Optional ('[').suppress () + Combine (Word (nums) + ZeroOrMore (oneOf (', -') + Word (nums)))
    volta_text = Literal ('[').suppress () + Regex (r'"[^"]+"')
    volta = volta_nums | volta_text
    invisible_barline = oneOf ('[|] []')
    dashed_barline = oneOf (': .|')
    double_rep = Literal (':') + FollowedBy (':')   # otherwise ambiguity with dashed barline
    voice_overlay = Combine (OneOrMore ('&'))
    bare_volta = FollowedBy (Literal ('[') + Word (nums))   # no barline, but volta follows (volta is parsed in next measure)
    bar_left = (oneOf ('[|: |: [: :') + Optional (volta)) | Optional ('|').suppress () + volta | oneOf ('| [|')
    bars = ZeroOrMore (':') + ZeroOrMore ('[') + OneOrMore (oneOf ('| ]'))
    bar_right = invisible_barline | double_rep | Combine (bars) | dashed_barline | voice_overlay | bare_volta
    
    errors =  ~bar_right + Optional (Word (' \n')) + CharsNotIn (':&|', exact=1)
    linebreak = Literal ('$') | ~decorations + Literal ('!')    # no need for I:linebreak !!!
    element = fld_or_lyr | broken | decorations | stem | chord_or_text | grace_notes | tuplet_start | linebreak | errors
    measure      = Group (ZeroOrMore (inline_field) + Optional (bar_left) + ZeroOrMore (element) + bar_right + Optional (linebreak) + Optional (lyr_blk))
    noBarMeasure = Group (ZeroOrMore (inline_field) + Optional (bar_left) + OneOrMore (element) + Optional (linebreak) + Optional (lyr_blk))
    abc_voice = ZeroOrMore (measure) + Optional (noBarMeasure | Group (bar_left)) + ZeroOrMore (inline_field).suppress () + StringEnd ()

    #----------------------------------------
    # I:percmap note [step] [midi] [note-head]
    #----------------------------------------

    white2 = (white | StringEnd ()).suppress ()
    w3 = Optional (white2)
    percid = Word (alphanums + '-')
    step = basenote + Optional (octave, 0)
    pitchg = Group (Optional (accidental, '') + step + FollowedBy (white2))
    stepg = Group (step + FollowedBy (white2)) | Literal ('*')
    midi = (Literal ('*') | number | pitchg | percid)
    nhd = Optional (Combine (percid + Optional ('+')), '')
    perc_wsp = Literal ('percmap') + w3 + pitchg + w3 + Optional (stepg, '*') + w3 + Optional (midi, '*') + w3 + nhd
    abc_percmap = perc_wsp.leaveWhitespace ()

    #----------------------------------------------------------------
    # Parse actions to convert all relevant results into an abstract
    # syntax tree where all tree nodes are instances of pObj
    #----------------------------------------------------------------

    ifield.setParseAction (lambda t: pObj ('field', t))
    grand_staff.setParseAction (lambda t: pObj ('grand', t, 1)) # 1 = keep ordered list of results
    brace_gr.setParseAction (lambda t: pObj ('bracegr', t, 1))
    bracket_gr.setParseAction (lambda t: pObj ('bracketgr', t, 1))
    voice_gr.setParseAction (lambda t: pObj ('voicegr', t, 1))
    voiceId.setParseAction (lambda t: pObj ('vid', t, 1))
    abc_scoredef.setParseAction (lambda t: pObj ('score', t, 1))
    note_length.setParseAction (lambda t: pObj ('dur', (t[0], (t[2] << len (t[1])) >> 1)))
    chordsym.setParseAction (lambda t: pObj ('chordsym', t))
    chord_root.setParseAction (lambda t: pObj ('root', t))
    chord_kind.setParseAction (lambda t: pObj ('kind', t))
    chord_degree.setParseAction (lambda t: pObj ('degree', t))
    chord_bass.setParseAction (lambda t: pObj ('bass', t))
    text_expression.setParseAction (lambda t: pObj ('text', t))
    inline_field.setParseAction (lambda t: pObj ('inline', t))
    lyr_fld.setParseAction (lambda t: pObj ('lyr_fld', t, 1))
    lyr_blk.setParseAction (lambda t: pObj ('lyr_blk', t, 1)) # 1 = keep ordered list of lyric lines
    grace_notes.setParseAction (doGrace)
    acciaccatura.setParseAction (lambda t: pObj ('accia', t))
    note.setParseAction (noteActn)
    rest.setParseAction (restActn)
    decorations.setParseAction (lambda t: pObj ('deco', t))
    pizzicato.setParseAction (lambda t: ['!plus!']) # translate !+!
    slur_ends.setParseAction (lambda t: pObj ('slurs', t))
    chord.setParseAction (lambda t: pObj ('chord', t, 1))
    dec_note.setParseAction (noteActn)
    tie.setParseAction (lambda t: pObj ('tie', t))
    pitch.setParseAction (lambda t: pObj ('pitch', t))
    bare_volta.setParseAction (lambda t: ['|']) # return barline that user forgot
    dashed_barline.setParseAction (lambda t: ['.|'])
    bar_right.setParseAction (lambda t: pObj ('rbar', t))
    bar_left.setParseAction (lambda t: pObj ('lbar', t))
    broken.setParseAction (lambda t: pObj ('broken', t))
    tuplet_start.setParseAction (lambda t: pObj ('tup', t))
    linebreak.setParseAction (lambda t: pObj ('linebrk', t))
    measure.setParseAction (doMaat)
    noBarMeasure.setParseAction (doMaat)
    b1.setParseAction (errorWarn)
    b2.setParseAction (errorWarn)
    b3.setParseAction (errorWarn)
    errors.setParseAction (errorWarn)

    return abc_header, abc_voice, abc_scoredef, abc_percmap

class pObj (object):    # every relevant parse result is converted into a pObj
    def __init__ (s, name, t, seq=0):   # t = list of nested parse results
        s.name = name   # name uniqueliy identifies this pObj
        rest = []       # collect parse results that are not a pObj
        attrs = {}      # new attributes
        for x in t:     # nested pObj's become attributes of this pObj
            if type (x) == pObj:
                attrs [x.name] = attrs.get (x.name, []) + [x]
            else:
                rest.append (x)             # collect non-pObj's (mostly literals)
        for name, xs in attrs.items ():
            if len (xs) == 1: xs = xs[0]    # only list if more then one pObj
            setattr (s, name, xs)           # create the new attributes
        s.t = rest      # all nested non-pObj's (mostly literals)
        s.objs = seq and t or []            # for nested ordered (lyric) pObj's

    def __repr__ (s):   # make a nice string representation of a pObj
        r = []
        for nm in dir (s):
            if nm.startswith ('_'): continue # skip build in attributes
            elif nm == 'name': continue     # redundant
            else:
                x = getattr (s, nm)
                if not x: continue          # s.t may be empty (list of non-pObj's)
                if type (x) == list_type:  r.extend (x)
                else:                           r.append (x)
        xs = []
        for x in r:     # recursively call __repr__ and convert all strings to latin-1
            if isinstance (x, str_type): xs.append (x)          # string -> no recursion
            else:                        xs.append (repr (x))   # pObj -> recursive call
        return '(' + s.name + ' ' +','.join (xs) + ')'

global prevloc                  # global to remember previous match position of a note/rest
prevloc = 0
def detectBeamBreak (line, loc, t):
    global prevloc              # location in string 'line' of previous note match
    xs = line[prevloc:loc+1]    # string between previous and current note match
    xs = xs.lstrip ()           # first note match starts on a space!
    prevloc = loc               # location in string 'line' of current note match
    b = pObj ('bbrk', [' ' in xs])      # space somewhere between two notes -> beambreak
    t.insert (0, b)             # insert beambreak as a nested parse result

def noteActn (line, loc, t):    # detect beambreak between previous and current note/rest
    if 'y' in t[0].t: return [] # discard spacer
    detectBeamBreak (line, loc, t)      # adds beambreak to parse result t as side effect
    return pObj ('note', t)

def restActn (line, loc, t):    # detect beambreak between previous and current note/rest
    detectBeamBreak (line, loc, t)  # adds beambreak to parse result t as side effect
    return pObj ('rest', t)

def errorWarn (line, loc, t):   # warning for misplaced symbols and skip them
    if not t[0]: return []      # only warn if catched string not empty
    info ('**misplaced symbol: %s' % t[0], warn=0)
    lineCopy = line [:]
    if loc > 40:
        lineCopy = line [loc - 40: loc + 40]
        loc = 40
    info (lineCopy.replace ('\n', ' '), warn=0)
    info (loc * '-' + '^', warn=0)
    return []

#-------------------------------------------------------------
# transformations of a measure (called by parse action doMaat)
#-------------------------------------------------------------

def simplify (a, b):    # divide a and b by their greatest common divisor
    x, y = a, b
    while b: a, b = b, a % b
    return x // a, y // a

def doBroken (prev, brk, x):
    if not prev: info ('error in broken rhythm: %s' % x); return    # no changes
    nom1, den1 = prev.dur.t # duration of first note/chord
    nom2, den2 = x.dur.t    # duration of second note/chord
    if  brk == '>':
        nom1, den1  = simplify (3 * nom1, 2 * den1)
        nom2, den2  = simplify (1 * nom2, 2 * den2)
    elif brk == '<':
        nom1, den1  = simplify (1 * nom1, 2 * den1)
        nom2, den2  = simplify (3 * nom2, 2 * den2)
    elif brk == '>>':
        nom1, den1  = simplify (7 * nom1, 4 * den1)
        nom2, den2  = simplify (1 * nom2, 4 * den2)
    elif brk == '<<':
        nom1, den1  = simplify (1 * nom1, 4 * den1)
        nom2, den2  = simplify (7 * nom2, 4 * den2)
    else: return            # give up
    prev.dur.t = nom1, den1 # change duration of previous note/chord
    x.dur.t = nom2, den2    # and current note/chord

def convertBroken (t):  # convert broken rhythms to normal note durations
    prev = None # the last note/chord before the broken symbol
    brk = ''    # the broken symbol
    remove = [] # indexes to broken symbols (to be deleted) in measure
    for i, x in enumerate (t):  # scan all elements in measure
        if x.name == 'note' or x.name == 'chord' or x.name == 'rest':
            if brk:                 # a broken symbol was encountered before
                doBroken (prev, brk, x) # change duration previous note/chord/rest and current one
                brk = ''
            else:
                prev = x            # remember the last note/chord/rest
        elif x.name == 'broken':
            brk = x.t[0]            # remember the broken symbol (=string)
            remove.insert (0, i)    # and its index, highest index first
    for i in remove: del t[i]       # delete broken symbols from high to low

def ptc2midi (n):       # convert parsed pitch attribute to a midi number
    pt = getattr (n, 'pitch', '')
    if pt:
        p = pt.t
        if len (p) == 3: acc, step, oct = p
        else:       acc = ''; step, oct = p
        nUp = step.upper ()
        oct = (4 if nUp == step else 5) + int (oct)
        midi = oct * 12 + [0,2,4,5,7,9,11]['CDEFGAB'.index (nUp)] + {'^':1,'_':-1}.get (acc, 0) + 12
    else: midi = 130    # all non pitch objects first
    return midi

def convertChord (t):   # convert chord to sequence of notes in musicXml-style
    ins = []
    for i, x in enumerate (t):
        if x.name == 'chord':
            if hasattr (x, 'rest') and not hasattr (x, 'note'): # chords containing only rests
                if type (x.rest) == list_type: x.rest = x.rest[0] # more rests == one rest
                ins.insert (0, (i, [x.rest]))   # just output a single rest, no chord
                continue
            num1, den1 = x.dur.t                # chord duration
            tie = getattr (x, 'tie', None)      # chord tie
            slurs = getattr (x, 'slurs', [])    # slur endings
            if type (x.note) != list_type: x.note = [x.note]    # when chord has only one note ...
            elms = []; j = 0                    # sort chord notes, highest first
            nss = sorted (x.objs, key = ptc2midi, reverse=1) if mxm.orderChords else x.objs
            for nt in nss:   # all chord elements (note | decorations | rest | grace note)
                if nt.name == 'note':
                    num2, den2 = nt.dur.t           # note duration * chord duration
                    nt.dur.t = simplify (num1 * num2, den1 * den2)
                    if tie: nt.tie = tie            # tie on all chord notes
                    if j == 0 and slurs: nt.slurs = slurs   # slur endings only on first chord note
                    if j > 0: nt.chord = pObj ('chord', [1]) # label all but first as chord notes
                    else:                           # remember all pitches of the chord in the first note
                        pitches = [n.pitch for n in x.note] # to implement conversion of erroneous ties to slurs
                        nt.pitches = pObj ('pitches', pitches)
                    j += 1
                if nt.name not in ['dur','tie','slurs','rest']: elms.append (nt)
            ins.insert (0, (i, elms))           # chord position, [note|decotation|grace note]
    for i, notes in ins:                        # insert from high to low
        for nt in reversed (notes):
            t.insert (i+1, nt)                  # insert chord notes after chord
        del t[i]                                # remove chord itself

def doMaat (t):             # t is a Group() result -> the measure is in t[0]
    convertBroken (t[0])    # remove all broken rhythms and convert to normal durations
    convertChord (t[0])     # replace chords by note sequences in musicXML style

def doGrace (t):        # t is a Group() result -> the grace sequence is in t[0]
    convertChord (t[0]) # a grace sequence may have chords
    for nt in t[0]:     # flag all notes within the grace sequence
        if nt.name == 'note': nt.grace = 1 # set grace attribute
    return t[0]         # ungroup the parse result
#--------------------
# musicXML generation
#----------------------------------

def compChordTab ():    # avoid some typing work: returns mapping constant {ABC chordsyms -> musicXML kind}
    maj, min, aug, dim, dom, ch7, ch6, ch9, ch11, ch13, hd = 'major minor augmented diminished dominant -seventh -sixth -ninth -11th -13th half-diminished'.split ()
    triad   = zip ('ma Maj maj M mi min m aug dim o + -'.split (), [maj, maj, maj, maj, min, min, min, aug, dim, dim, aug, min])
    seventh = zip ('7 ma7 Maj7 M7 maj7 mi7 min7 m7 dim7 o7 -7 aug7 +7 m7b5 mi7b5'.split (),
                   [dom, maj+ch7, maj+ch7, maj+ch7, maj+ch7, min+ch7, min+ch7, min+ch7, dim+ch7, dim+ch7, min+ch7, aug+ch7, aug+ch7, hd, hd])
    sixth   = zip ('6 ma6 M6 mi6 min6 m6'.split (), [maj+ch6, maj+ch6, maj+ch6, min+ch6, min+ch6, min+ch6])
    ninth   = zip ('9 ma9 M9 maj9 Maj9 mi9 min9 m9'.split (), [dom+ch9, maj+ch9, maj+ch9, maj+ch9, maj+ch9, min+ch9, min+ch9, min+ch9])
    elevn   = zip ('11 ma11 M11 maj11 Maj11 mi11 min11 m11'.split (), [dom+ch11, maj+ch11, maj+ch11, maj+ch11, maj+ch11, min+ch11, min+ch11, min+ch11])
    thirt   = zip ('13 ma13 M13 maj13 Maj13 mi13 min13 m13'.split (), [dom+ch13, maj+ch13, maj+ch13, maj+ch13, maj+ch13, min+ch13, min+ch13, min+ch13])
    sus     = zip ('sus sus4 sus2'.split (), ['suspended-fourth', 'suspended-fourth', 'suspended-second'])
    return dict (list (triad) + list (seventh) + list (sixth) + list (ninth) + list (elevn) + list (thirt) + list (sus))

def addElem (parent, child, level):
    indent = 2
    chldrn = list (parent)
    if chldrn:
        chldrn[-1].tail += indent * ' '
    else:
        parent.text = '\n' + level * indent * ' '
    parent.append (child)
    child.tail = '\n' + (level-1) * indent * ' '

def addElemT (parent, tag, text, level):
    e = E.Element (tag)
    e.text = text
    addElem (parent, e, level)
    return e
    
def mkTmod (tmnum, tmden, lev):
    tmod = E.Element ('time-modification')
    addElemT (tmod, 'actual-notes', str (tmnum), lev + 1)
    addElemT (tmod, 'normal-notes', str (tmden), lev + 1)
    return tmod

def addDirection (parent, elems, lev, gstaff, subelms=[], placement='below', cue_on=0):
    dir = E.Element ('direction', placement=placement)
    addElem (parent, dir, lev)
    if type (elems) != list_type: elems = [(elems, subelms)]    # ugly hack to provide for multiple direction types
    for elem, subelms in elems: # add direction types
        typ = E.Element ('direction-type')
        addElem (dir, typ, lev + 1)
        addElem (typ, elem, lev + 2)
        for subel in subelms: addElem (elem, subel, lev + 3)
    if cue_on: addElem (dir, E.Element ('level', size='cue'), lev + 1)
    if gstaff: addElemT (dir, 'staff', str (gstaff), lev + 1)
    return dir

def removeElems (root_elem, parent_str, elem_str):
    for p in root_elem.findall (parent_str):
        e = p.find (elem_str)
        if e != None: p.remove (e)

def alignLyr (vce, lyrs):
    empty_el = pObj ('leeg', '*')
    for k, lyr in enumerate (lyrs): # lyr = one full line of lyrics
        i = 0               # syl counter
        for elem in vce:    # reiterate the voice block for each lyrics line
            if elem.name == 'note' and not (hasattr (elem, 'chord') or hasattr (elem, 'grace')):
                if i >= len (lyr): lr = empty_el
                else: lr = lyr [i]
                lr.t[0] = lr.t[0].replace ('%5d',']')
                elem.objs.append (lr)
                if lr.name != 'sbar': i += 1
            if elem.name == 'rbar' and i < len (lyr) and lyr[i].name == 'sbar': i += 1
    return vce

slur_move = re.compile (r'(?<![!+])([}><][<>]?)(\)+)')  # (?<!...) means: not preceeded by ...
mm_rest = re.compile (r'([XZ])(\d+)')
bar_space = re.compile (r'([:|][ |\[\]]+[:|])')         # barlines with spaces
def fixSlurs (x):   # repair slurs when after broken sign or grace-close
    def f (mo):     # replace a multi-measure rest by single measure rests
        n = int (mo.group (2))
        return (n * (mo.group (1) + '|')) [:-1]
    def g (mo):     # squash spaces in barline expressions
        return mo.group (1).replace (' ','')
    x = mm_rest.sub (f, x)
    x = bar_space.sub (g, x)
    return slur_move.sub (r'\2\1', x)

def splitHeaderVoices (abctext):
    escField = lambda x: '[' + x.replace (']',r'%5d') + ']' # hope nobody uses %5d in a field
    r1 = re.compile (r'%.*$')           # comments
    r2 = re.compile (r'^([A-Zw]:.*$)|\[[A-Zw]:[^]]*]$')     # information field, including lyrics
    r3 = re.compile (r'^%%(?=[^%])')    # directive: ^%% folowed by not a %
    xs, nx, mcont, fcont = [], 0, 0, 0  # result lines, X-encountered, music continuation, field continuation
    mln = fln = ''                      # music line, field line
    for x in abctext.splitlines ():
        x = x.strip ()
        if not x and nx == 1: break     # end of tune (empty line)
        if x.startswith ('X:'):
            if nx == 1: break           # second tune starts without an empty line !!
            nx = 1                      # start first tune
        x = r3.sub ('I:', x)            # replace %% -> I:
        x2 = r1.sub ('', x)             # remove comment
        while x2.endswith ('*') and not (x2.startswith ('w:') or x2.startswith ('+:') or 'percmap' in x2):
            x2 = x2[:-1]                # remove old syntax for right adjusting
        if not x2: continue             # empty line
        if x2[:2] == 'W:':
            field = x2 [2:].strip ()
            ftype = mxm.metaMap.get ('W', 'W')  # respect the (user defined --meta) mapping of various ABC fields to XML meta data types
            c = mxm.metadata.get (ftype, '')
            mxm.metadata [ftype] = c + '\n' + field if c else field   # concatenate multiple info fields with new line as separator
            continue                    # skip W: lyrics
        if x2[:2] == '+:':              # field continuation
            fln += x2[2:]
            continue
        ro = r2.match (x2)              # single field on a line
        if ro:                          # field -> inline_field, escape all ']'
            if fcont:                   # old style \-info-continuation active
                fcont = x2 [-1] == '\\' # possible further \-info-continuation
                fln += re.sub (r'^.:(.*?)\\*$', r'\1', x2) # add continuation, remove .: and \
                continue
            if fln: mln += escField (fln)
            if x2.startswith ('['): x2 = x2.strip ('[]')
            fcont = x2 [-1] == '\\'     # first encounter of old style \-info-continuation
            fln = x2.rstrip ('\\')      # remove continuation from field and inline brackets
            continue
        if nx == 1:                     # x2 is a new music line
            fcont = 0                   # stop \-continuations (-> only adjacent \-info-continuations are joined)
            if fln:
                mln +=  escField (fln)
                fln = ''
            if mcont:
                mcont = x2 [-1] == '\\' 
                mln += x2.rstrip ('\\')
            else:
                if mln: xs.append (mln); mln = ''
                mcont = x2 [-1] == '\\'
                mln = x2.rstrip ('\\')
            if not mcont: xs.append (mln); mln = ''
    if fln: mln += escField (fln)
    if mln: xs.append (mln)

    hs = re.split (r'(\[K:[^]]*\])', xs [0])   # look for end of header K:
    if len (hs) == 1: header = hs[0]; xs [0] = ''               # no K: present
    else: header = hs [0] + hs [1]; xs [0] = ''.join (hs[2:])   # h[1] is the first K:
    abctext = '\n'.join (xs)                    # the rest is body text
    hfs, vfs = [], []
    for x in header[1:-1].split (']['):
        if x[0] == 'V': vfs.append (x)          # filter voice- and midi-definitions
        elif x[:6] == 'I:MIDI': vfs.append (x)  # from the header to vfs
        elif x[:9] == 'I:percmap': vfs.append (x)  # and also percmap
        else: hfs.append (x)                    # all other fields stay in header
    header = '[' + ']['.join (hfs) + ']'        # restore the header
    abctext = ('[' + ']['.join (vfs) + ']' if vfs else '') + abctext    # prepend voice/midi from header before abctext

    xs = abctext.split ('[V:')
    if len (xs) == 1: abctext = '[V:1]' + abctext # abc has no voice defs at all
    elif re.sub (r'\[[A-Z]:[^]]*\]', '', xs[0]).strip ():   # remove inline fields from starting text, if any
        abctext = '[V:1]' + abctext     # abc with voices has no V: at start

    r1 = re.compile (r'\[V:\s*(\S*)[ \]]') # get voice id from V: field (skip spaces betwee V: and ID)
    vmap = {}                           # {voice id -> [voice abc string]}
    vorder = {}                         # mark document order of voices
    xs = re.split (r'(\[V:[^]]*\])', abctext)   # split on every V-field (V-fields included in split result list)
    if len (xs) == 1: raise ValueError ('bugs ...')
    else:
        pm = re.findall (r'\[P:.\]', xs[0])         # all P:-marks after K: but before first V:
        if pm: xs[2] = ''.join (pm) + xs[2]         # prepend P:-marks to the text of the first voice
        header += re.sub (r'\[P:.\]', '', xs[0])    # clear all P:-marks from text between K: and first V: and put text in the header
        i = 1
        while i < len (xs):             # xs = ['', V-field, voice abc, V-field, voice abc, ...]
            vce, abc = xs[i:i+2]
            id = r1.search (vce).group (1)                  # get voice ID from V-field
            if not id: id, vce = '1', '[V:1]'               # voice def has no ID
            vmap[id] = vmap.get (id, []) + [vce, abc]       # collect abc-text for each voice id (include V-fields)
            if id not in vorder: vorder [id] = i            # store document order of first occurrence of voice id
            i += 2
    voices = []
    ixs = sorted ([(i, id) for id, i in vorder.items ()])   # restore document order of voices
    for i, id in ixs:
        voice = ''.join (vmap [id])     # all abc of one voice
        voice = fixSlurs (voice)        # put slurs right after the notes
        voices.append ((id, voice))
    return header, voices

def mergeMeasure (m1, m2, slur_offset, voice_offset, rOpt, is_grand=0, is_overlay=0):
    slurs = m2.findall ('note/notations/slur')
    for slr in slurs:
        slrnum = int (slr.get ('number')) + slur_offset 
        slr.set ('number', str (slrnum))    # make unique slurnums in m2
    vs = m2.findall ('note/voice')          # set all voice number elements in m2
    for v in vs: v.text  = str (voice_offset + int (v.text))
    ls = m1.findall ('note/lyric')          # all lyric elements in m1
    lnum_max = max ([int (l.get ('number')) for l in ls] + [0]) # highest lyric number in m1
    ls = m2.findall ('note/lyric')          # update lyric elements in m2
    for el in ls:
        n = int (el.get ('number'))
        el.set ('number', str (n + lnum_max))
    ns = m1.findall ('note')    # determine the total duration of m1, subtract all backups
    dur1 = sum (int (n.find ('duration').text) for n in ns
                if n.find ('grace') == None and n.find ('chord') == None)
    dur1 -= sum (int (b.text) for b in m1.findall ('backup/duration'))
    repbar, nns, es = 0, 0, []  # nns = number of real notes in m2
    for e in list (m2): # scan all elements of m2
        if e.tag == 'attributes':
            if not is_grand: continue # no attribute merging for normal voices
            else: nns += 1      # but we do merge (clef) attributes for a grand staff
        if e.tag == 'print': continue
        if e.tag == 'note' and (rOpt or e.find ('rest') == None): nns += 1
        if e.tag == 'barline' and e.find ('repeat') != None: repbar = e;
        es.append (e)           # buffer elements to be merged
    if nns > 0:                 # only merge if m2 contains any real notes
        if dur1 > 0:            # only insert backup if duration of m1 > 0
            b = E.Element ('backup')
            addElem (m1, b, level=3)
            addElemT (b, 'duration', str (dur1), level=4)
        for e in es: addElem (m1, e, level=3)   # merge buffered elements of m2
    elif is_overlay and repbar: addElem (m1, repbar, level=3)   # merge repeat in empty overlay

def mergePartList (parts, rOpt, is_grand=0):    # merge parts, make grand staff when is_grand true

    def delAttrs (part):                # for the time being we only keep clef attributes
        xs = [(m, e) for m in part.findall ('measure') for e in m.findall ('attributes')]
        for m, e in xs:
            for c in list (e):
                if c.tag == 'clef': continue    # keep clef attribute
                if c.tag == 'staff-details': continue    # keep staff-details attribute
                e.remove (c)                    # delete all other attrinutes for higher staff numbers
            if len (list (e)) == 0: m.remove (e)    # remove empty attributes element

    p1 = parts[0]
    for p2 in parts[1:]:
        if is_grand: delAttrs (p2)                          # delete all attributes except clef
        for i in range (len (p1) + 1, len (p2) + 1):        # second part longer than first one
            maat = E.Element ('measure', number = str(i))   # append empty measures
            addElem (p1, maat, 2)
        slurs = p1.findall ('measure/note/notations/slur')  # find highest slur num in first part
        slur_max = max ([int (slr.get ('number')) for slr in slurs] + [0])
        vs = p1.findall ('measure/note/voice')              # all voice number elements in first part
        vnum_max = max ([int (v.text) for v in vs] + [0])   # highest voice number in first part
        for im, m2 in enumerate (p2.findall ('measure')):   # merge all measures of p2 into p1
            mergeMeasure (p1[im], m2, slur_max, vnum_max, rOpt, is_grand) # may change slur numbers in p1
    return p1

def mergeParts (parts, vids, staves, rOpt, is_grand=0):
    if not staves: return parts, vids   # no voice mapping
    partsnew, vidsnew = [], []
    for voice_ids in staves:
        pixs = []
        for vid in voice_ids:
            if vid in vids: pixs.append (vids.index (vid))
            else: info ('score partname %s does not exist' % vid)
        if pixs:
            xparts = [parts[pix] for pix in pixs]
            if len (xparts) > 1: mergedpart = mergePartList (xparts, rOpt, is_grand)
            else:                mergedpart = xparts [0]
            partsnew.append (mergedpart)
            vidsnew.append (vids [pixs[0]])
    return partsnew, vidsnew

def mergePartMeasure (part, msre, ovrlaynum, rOpt): # merge msre into last measure of part, only for overlays
    slur_offset = 0;    # slur numbers determined by the slurstack size (as in a single voice)
    last_msre = list (part)[-1] # last measure in part
    mergeMeasure (last_msre, msre, slur_offset, ovrlaynum, rOpt, is_overlay=1) # voice offset = s.overlayVNum

def pushSlur (boogStapel, stem):
    if stem not in boogStapel: boogStapel [stem] = [] # initialize slurstack for stem
    boognum = sum (map (len, boogStapel.values ())) + 1  # number of open slurs in all (overlay) voices
    boogStapel [stem].append (boognum)
    return boognum

def setFristVoiceNameFromGroup (vids, vdefs): # vids = [vid], vdef = {vid -> (name, subname, voicedef)}
    vids = [v for v in vids if v in vdefs]  # only consider defined voices
    if not vids: return vdefs
    vid0 = vids [0]                         # first vid of the group
    _, _, vdef0 = vdefs [vid0]              # keep de voice definition (vdef0) when renaming vid0
    for vid in vids:
        nm, snm, vdef = vdefs [vid]
        if nm:                              # first non empty name encountered will become
            vdefs [vid0] = nm, snm, vdef0   # name of merged group == name of first voice in group (vid0)
            break
    return vdefs

def mkGrand (p, vdefs):             # transform parse subtree into list needed for s.grands
    xs = []
    for i, x in enumerate (p.objs): # changing p.objs [i] alters the tree. changing x has no effect on the tree.
        if type (x) == pObj:
            us = mkGrand (x, vdefs) # first get transformation results of current pObj
            if x.name == 'grand':   # x.objs contains ordered list of nested parse results within x
                vids = [y.objs[0] for y in x.objs[1:]]  # the voice ids in the grand staff
                nms = [vdefs [u][0] for u in vids if u in vdefs] # the names of those voices
                accept = sum ([1 for nm in nms if nm]) == 1 # accept as grand staff when only one of the voices has a name
                if accept or us[0] == '{*':
                    xs.append (us[1:])      # append voice ids as a list (discard first item '{' or '{*')
                    vdefs = setFristVoiceNameFromGroup (vids, vdefs)
                    p.objs [i] = x.objs[1]  # replace voices by first one in the grand group (this modifies the parse tree)
                else:
                    xs.extend (us[1:])      # extend current result with all voice ids of rejected grand staff
            else: xs.extend (us)    # extend current result with transformed pObj
        else: xs.append (p.t[0])    # append the non pObj (== voice id string)
    return xs

def mkStaves (p, vdefs):            # transform parse tree into list needed for s.staves
    xs = []
    for i, x in enumerate (p.objs): # structure and comments identical to mkGrand
        if type (x) == pObj:
            us = mkStaves (x, vdefs)
            if x.name == 'voicegr':
                xs.append (us)
                vids = [y.objs[0] for y in x.objs]
                vdefs = setFristVoiceNameFromGroup (vids, vdefs)
                p.objs [i] = x.objs[0]
            else:
                xs.extend (us)
        else:
            if p.t[0] not in '{*':  xs.append (p.t[0])
    return xs

def mkGroups (p):                   # transform parse tree into list needed for s.groups
    xs = []
    for x in p.objs:
        if type (x) == pObj:
            if x.name == 'vid': xs.extend (mkGroups (x))
            elif x.name == 'bracketgr': xs.extend (['['] + mkGroups (x) + [']'])
            elif x.name == 'bracegr':   xs.extend (['{'] + mkGroups (x) + ['}'])
            else: xs.extend (mkGroups (x) + ['}'])  # x.name == 'grand' == rejected grand staff
        else:
            xs.append (p.t[0])
    return xs

def stepTrans (step, soct, clef):   # [A-G] (1...8)
    if clef.startswith ('bass'):
        nm7 = 'C,D,E,F,G,A,B'.split (',')
        n = 14 + nm7.index (step) - 12  # two octaves extra to avoid negative numbers
        step, soct = nm7 [n % 7], soct + n // 7 - 2  # subtract two octaves again
    return step, soct

def reduceMids (parts, vidsnew, midiInst):       # remove redundant instruments from a part
    for pid, part in zip (vidsnew, parts):
        mids, repls, has_perc = {}, {}, 0
        for ipid, ivid, ch, prg, vol, pan in sorted (list (midiInst.values ())):
            if ipid != pid: continue                # only instruments from part pid
            if ch == '10': has_perc = 1; continue   # only consider non percussion instruments
            instId, inst = 'I%s-%s' % (ipid, ivid), (ch, prg)
            if inst in mids:                        # midi instrument already defined in this part
                repls [instId]  = mids [inst]       # remember to replace instId by inst (see below)
                del midiInst [instId]               # instId is redundant
            else: mids [inst] = instId              # collect unique instruments in this part
        if len (mids) < 2 and not has_perc:         # only one instrument used -> no instrument tags needed in notes
            removeElems (part, 'measure/note', 'instrument')    # no instrument tag needed for one- or no-instrument parts
        else:
            for e in part.findall ('measure/note/instrument'):
                id = e.get ('id')                   # replace all redundant instrument Id's
                if id in repls: e.set ('id', repls [id])

class stringAlloc:
    def __init__ (s):
        s.snaarVrij = []    # [[(t1, t2) ...] for each string ]
        s.snaarIx = []      # index in snaarVrij for each string
        s.curstaff = -1     # staff being allocated
    def beginZoek (s):      # reset snaarIx at start of each voice
        s.snaarIx = []
        for i in range (len (s.snaarVrij)): s.snaarIx.append (0)
    def setlines (s, stflines, stfnum):
        if stfnum != s.curstaff:    # initialize for new staff
            s.curstaff = stfnum
            s.snaarVrij = []
            for i in range (stflines): s.snaarVrij.append ([])
            s.beginZoek ()
    def isVrij (s, snaar, t1, t2):  # see if string snaar is free between t1 and t2
        xs = s.snaarVrij [snaar]
        for i in range (s.snaarIx [snaar], len (xs)):
            tb, te = xs [i]
            if t1 >= te: continue   # te_prev < t1 <= te
            if t1 >= tb: s.snaarIx [snaar] = i; return 0    # tb <= t1 < te
            if t2 > tb: s.snaarIx [snaar] = i; return 0     # t1 < tb < t2
            s.snaarIx [snaar] = i;  # remember position for next call
            xs.insert (i, (t1,t2))  # te_prev < t1 < t2 < tb
            return 1
        xs.append ((t1,t2))
        s.snaarIx [snaar] = len (xs) - 1
        return 1
    def bezet (s, snaar, t1, t2):   # force allocation of note (t1,t2) on string snaar
        xs = s.snaarVrij [snaar]
        for i, (tb, te) in enumerate (xs):
            if t1 >= te: continue   # te_prev < t1 <= te
            xs.insert (i, (t1, t2))
            return
        xs.append ((t1,t2))

class MusicXml:
    typeMap = {1:'long', 2:'breve', 4:'whole', 8:'half', 16:'quarter', 32:'eighth', 64:'16th', 128:'32nd', 256:'64th'}
    dynaMap = {'p':1,'pp':1,'ppp':1,'pppp':1,'f':1,'ff':1,'fff':1,'ffff':1,'mp':1,'mf':1,'sfz':1}
    tempoMap = {'larghissimo':40, 'moderato':104, 'adagissimo':44, 'allegretto':112, 'lentissimo':48, 'allegro':120, 'largo':56,
            'vivace':168, 'adagio':59, 'vivo':180, 'lento':62, 'presto':192, 'larghetto':66, 'allegrissimo':208, 'adagietto':76,
            'vivacissimo':220, 'andante':88, 'prestissimo':240, 'andantino':96}
    wedgeMap = {'>(':1, '>)':1, '<(':1,'<)':1,'crescendo(':1,'crescendo)':1,'diminuendo(':1,'diminuendo)':1}
    artMap = {'.':'staccato','>':'accent','accent':'accent','wedge':'staccatissimo','tenuto':'tenuto',
              'breath':'breath-mark','marcato':'strong-accent','^':'strong-accent','slide':'scoop'}
    ornMap = {'trill':'trill-mark','T':'trill-mark','turn':'turn','uppermordent':'inverted-mordent','lowermordent':'mordent',
              'pralltriller':'inverted-mordent','mordent':'mordent','turn':'turn','invertedturn':'inverted-turn'}
    tecMap = {'upbow':'up-bow', 'downbow':'down-bow', 'plus':'stopped','open':'open-string','snap':'snap-pizzicato',
              'thumb':'thumb-position'}
    capoMap = {'fine':('Fine','fine','yes'), 'D.S.':('D.S.','dalsegno','segno'), 'D.C.':('D.C.','dacapo','yes'),'dacapo':('D.C.','dacapo','yes'),
               'dacoda':('To Coda','tocoda','coda'), 'coda':('coda','coda','coda'), 'segno':('segno','segno','segno')}
    sharpness = ['Fb', 'Cb','Gb','Db','Ab','Eb','Bb','F','C','G','D','A', 'E', 'B', 'F#','C#','G#','D#','A#','E#','B#']
    offTab = {'maj':8, 'm':11, 'min':11, 'mix':9, 'dor':10, 'phr':12, 'lyd':7, 'loc':13}
    modTab = {'maj':'major', 'm':'minor', 'min':'minor', 'mix':'mixolydian', 'dor':'dorian', 'phr':'phrygian', 'lyd':'lydian', 'loc':'locrian'}
    clefMap = { 'alto1':('C','1'), 'alto2':('C','2'), 'alto':('C','3'), 'alto4':('C','4'), 'tenor':('C','4'),
                'bass3':('F','3'), 'bass':('F','4'), 'treble':('G','2'), 'perc':('percussion',''), 'none':('',''), 'tab':('TAB','5')}
    clefLineMap = {'B':'treble', 'G':'alto1', 'E':'alto2', 'C':'alto', 'A':'tenor', 'F':'bass3', 'D':'bass'}
    alterTab = {'=':'0', '_':'-1', '__':'-2', '^':'1', '^^':'2'}
    accTab = {'=':'natural', '_':'flat', '__':'flat-flat', '^':'sharp', '^^':'sharp-sharp'}
    chordTab = compChordTab ()
    uSyms = {'~':'roll', 'H':'fermata','L':'>','M':'lowermordent','O':'coda',
             'P':'uppermordent','S':'segno','T':'trill','u':'upbow','v':'downbow'}
    pageFmtDef = [0.75,297,210,18,18,10,10] # the abcm2ps page formatting defaults for A4
    metaTab = {'O':'origin', 'A':'area', 'Z':'transcription', 'N':'notes', 'G':'group', 'H':'history', 'R':'rhythm',
                'B':'book', 'D':'discography', 'F':'fileurl', 'S':'source', 'P':'partmap', 'W':'lyrics'}
    metaMap = {'C':'composer'}  # mapping of composer is fixed
    metaTypes = {'composer':1,'lyricist':1,'poet':1,'arranger':1,'translator':1, 'rights':1} # valid MusicXML meta data types
    tuningDef = 'E2,A2,D3,G3,B3,E4'.split (',') # default string tuning (guitar)

    def __init__ (s):
        s.pageFmtCmd = []   # set by command line option -p
        s.reset ()
    def reset (s, fOpt=False):
        s.divisions = 2520  # xml duration of 1/4 note, 2^3 * 3^2 * 5 * 7 => 5,7,9 tuplets
        s.ties = {}         # {abc pitch tuple -> alteration} for all open ties
        s.slurstack = {}    # stack of open slur numbers per (overlay) voice
        s.slurbeg = []      # type of slurs to start (when slurs are detected at element-level)
        s.tmnum = 0         # time modification, numerator
        s.tmden = 0         # time modification, denominator
        s.ntup = 0          # number of tuplet notes remaining
        s.trem = 0          # number of bars for tremolo
        s.intrem = 0        # mark tremolo sequence (for duration doubling)
        s.tupnts = []       # all tuplet modifiers with corresp. durations: [(duration, modifier), ...]
        s.irrtup = 0        # 1 if an irregular tuplet
        s.ntype = ''        # the normal-type of a tuplet (== duration type of a normal tuplet note)
        s.unitL =  (1, 8)   # default unit length
        s.unitLcur = (1, 8) # unit length of current voice
        s.keyAlts = {}      # alterations implied by key
        s.msreAlts = {}     # temporarily alterations
        s.curVolta = ''     # open volta bracket
        s.title = ''        # title of music
        s.creator = {}      # {creator-type -> creator string}
        s.metadata = {}     # {metadata-type -> string}
        s.lyrdash = {}      # {lyric number -> 1 if dash between syllables}
        s.usrSyms = s.uSyms # user defined symbols
        s.prevNote = None   # xml element of previous beamed note to correct beams (start, continue)
        s.prevLyric = {}    # xml element of previous lyric to add/correct extend type (start, continue)
        s.grcbbrk = False   # remember any bbrk in a grace sequence
        s.linebrk = 0       # 1 if next measure should start with a line break
        s.nextdecos = []    # decorations for the next note
        s.prevmsre = None   # the previous measure
        s.supports_tag = 0  # issue supports-tag in xml file when abc uses explicit linebreaks
        s.staveDefs = []    # collected %%staves or %%score instructions from score
        s.staves = []       # staves = [[voice names to be merged into one stave]]
        s.groups = []       # list of merged part names with interspersed {[ and }]
        s.grands = []       # [[vid1, vid2, ..], ...] voiceIds to be merged in a grand staff
        s.gStaffNums = {}   # map each voice id in a grand staff to a staff number
        s.gNstaves = {}     # map each voice id in a grand staff to total number of staves
        s.pageFmtAbc = []   # formatting from abc directives
        s.mdur = (4,4)      # duration of one measure
        s.gtrans = 0        # octave transposition (by clef)
        s.midprg = ['', '', '', ''] # MIDI channel nr, program nr, volume, panning for the current part
        s.vid = ''          # abc voice id for the current voice
        s.pid = ''          # xml part id for the current voice
        s.gcue_on = 0       # insert <cue/> tag in each note
        s.percVoice = 0     # 1 if percussion enabled
        s.percMap = {}      # (part-id, abc_pitch, xml-octave) -> (abc staff step, midi note number, xml notehead)
        s.pMapFound = 0     # at least one I:percmap has been found
        s.vcepid = {}       # voice_id -> part_id
        s.midiInst = {}     # inst_id -> (part_id, voice_id, channel, midi_number), remember instruments used
        s.capo = 0          # fret position of the capodastro
        s.tunmid = []       # midi numbers of strings
        s.tunTup = []       # ordered midi numbers of strings [(midi_num, string_num), ...] (midi_num from high to low)
        s.fOpt = fOpt       # force string/fret allocations for tab staves
        s.orderChords = 0   # order notes in a chord
        s.chordDecos = {}   # decos that should be distributed to all chord notes for xml
        ch10 = 'acoustic-bass-drum,35;bass-drum-1,36;side-stick,37;acoustic-snare,38;hand-clap,39;electric-snare,40;low-floor-tom,41;closed-hi-hat,42;high-floor-tom,43;pedal-hi-hat,44;low-tom,45;open-hi-hat,46;low-mid-tom,47;hi-mid-tom,48;crash-cymbal-1,49;high-tom,50;ride-cymbal-1,51;chinese-cymbal,52;ride-bell,53;tambourine,54;splash-cymbal,55;cowbell,56;crash-cymbal-2,57;vibraslap,58;ride-cymbal-2,59;hi-bongo,60;low-bongo,61;mute-hi-conga,62;open-hi-conga,63;low-conga,64;high-timbale,65;low-timbale,66;high-agogo,67;low-agogo,68;cabasa,69;maracas,70;short-whistle,71;long-whistle,72;short-guiro,73;long-guiro,74;claves,75;hi-wood-block,76;low-wood-block,77;mute-cuica,78;open-cuica,79;mute-triangle,80;open-triangle,81'
        s.percsnd = [x.split (',') for x in ch10.split (';')]   # {name -> midi number} of standard channel 10 sound names
        s.gTime = (0,0)     # (XML begin time, XML end time) in divisions
        s.tabStaff = ''     # == pid (part ID) for a tab staff

    def mkPitch (s, acc, note, oct, lev):
        if s.percVoice: # percussion map switched off by perc=off (see doClef)
            octq = int (oct) + s.gtrans     # honour the octave= transposition when querying percmap
            tup = s.percMap.get ((s.pid, acc+note, octq), s.percMap.get (('', acc+note, octq), 0))
            if tup: step, soct, midi, notehead = tup
            else: step, soct = note, octq
            octnum = (4 if step.upper() == step else 5) + int (soct)
            if not tup: # add percussion map for unmapped notes in this part
                midi = str (octnum * 12 + [0,2,4,5,7,9,11]['CDEFGAB'.index (step.upper())] + {'^':1,'_':-1}.get (acc, 0) + 12)
                notehead = {'^':'x', '_':'circle-x'}.get (acc, 'normal')
                if s.pMapFound: info ('no I:percmap for: %s%s in part %s, voice %s' % (acc+note, -oct*',' if oct<0 else oct*"'", s.pid, s.vid))
                s.percMap [(s.pid, acc+note, octq)] =  (note, octq, midi, notehead)
            else:       # correct step value for clef
                step, octnum = stepTrans (step.upper (), octnum, s.curClef)
            pitch = E.Element ('unpitched')
            addElemT (pitch, 'display-step', step.upper (), lev + 1)
            addElemT (pitch, 'display-octave', str (octnum), lev + 1)
            return pitch, '', midi, notehead
        nUp = note.upper ()
        octnum = (4 if nUp == note else 5) + int (oct) + s.gtrans
        pitch = E.Element ('pitch')
        addElemT (pitch, 'step', nUp, lev + 1)
        alter = ''
        if (note, oct) in s.ties:
            tied_alter, _, vnum, _ = s.ties [(note,oct)]            # vnum = overlay voice number when tie started
            if vnum == s.overlayVnum: alter = tied_alter            # tied note in the same overlay -> same alteration
        elif acc:
            s.msreAlts [(nUp, octnum)] = s.alterTab [acc]
            alter = s.alterTab [acc]                                # explicit notated alteration
        elif (nUp, octnum) in s.msreAlts:   alter = s.msreAlts [(nUp, octnum)]  # temporary alteration
        elif nUp in s.keyAlts:              alter = s.keyAlts [nUp] # alteration implied by the key
        if alter: addElemT (pitch, 'alter', alter, lev + 1)
        addElemT (pitch, 'octave', str (octnum), lev + 1)
        return pitch, alter, '', ''

    def getNoteDecos (s, n):
        decos = s.nextdecos             # decorations encountered so far
        ndeco = getattr (n, 'deco', 0)  # possible decorations of notes of a chord
        if ndeco:                       # add decorations, translate used defined symbols
            decos += [s.usrSyms.get (d, d).strip ('!+') for d in ndeco.t]
        s.nextdecos = []
        if s.tabStaff == s.pid and s.fOpt and n.name != 'rest':  # force fret/string allocation if explicit string decoration is missing
            if [d for d in decos if d in '0123456789'] == []: decos.append ('0')
        return decos

    def mkNote (s, n, lev):
        isgrace = getattr (n, 'grace', '')
        ischord = getattr (n, 'chord', '')
        if s.ntup >= 0 and not isgrace and not ischord:
            s.ntup -= 1                 # count tuplet notes only on non-chord, non grace notes
            if s.ntup == -1 and s.trem <= 0:
                s.intrem = 0            # tremolo pair ends at first note that is not a new tremolo pair (s.trem > 0)
        nnum, nden = n.dur.t            # abc dutation of note
        if s.intrem: nnum += nnum       # double duration of tremolo duplets
        if nden == 0: nden = 1          # occurs with illegal ABC like: "A2 1". Now interpreted as A2/1
        num, den = simplify (nnum * s.unitLcur[0], nden * s.unitLcur[1])  # normalised with unit length
        if den > 64:    # limit denominator to 64
            num = int (round (64 * float (num) / den))  # scale note to num/64
            num, den  = simplify (max ([num, 1]), 64)   # smallest num == 1
            info ('duration too small: rounded to %d/%d' % (num, den))
        if n.name == 'rest' and ('Z' in n.t or 'X' in n.t):
              num, den = s.mdur         # duration of one measure
        noMsrRest = not (n.name == 'rest' and (num, den) == s.mdur) # not a measure rest
        dvs = (4 * s.divisions * num) // den    # divisions is xml-duration of 1/4
        rdvs = dvs                      # real duration (will be 0 for chord/grace)
        num, den = simplify (num, den * 4)      # scale by 1/4 for s.typeMap
        ndot = 0
        if num == 3 and noMsrRest: ndot = 1; den = den // 2 # look for dotted notes
        if num == 7 and noMsrRest: ndot = 2; den = den // 4
        nt = E.Element ('note')
        if isgrace:                     # a grace note (and possibly a chord note)
            grace = E.Element ('grace')
            if s.acciatura: grace.set ('slash', 'yes'); s.acciatura = 0
            addElem (nt, grace, lev + 1)
            dvs = rdvs = 0              # no (real) duration for a grace note
            if den <= 16: den = 32      # not longer than 1/8 for a grace note
        if s.gcue_on:                   # insert cue tag
            cue = E.Element ('cue')
            addElem (nt, cue, lev + 1)
        if ischord:                     # a chord note
            chord = E.Element ('chord')
            addElem (nt, chord, lev + 1)
            rdvs = 0                    # chord notes no real duration
        if den not in s.typeMap:        # take the nearest smaller legal duration
            info ('illegal duration %d/%d' % (nnum, nden))
            den = min (x for x in s.typeMap.keys () if x > den)
        xmltype = str (s.typeMap [den]) # xml needs the note type in addition to duration
        acc, step, oct = '', 'C', '0'   # abc-notated pitch elements (accidental, pitch step, octave)
        alter, midi, notehead = '', '', ''      # xml alteration
        if n.name == 'rest':
            if 'x' in n.t or 'X' in n.t: nt.set ('print-object', 'no')
            rest = E.Element ('rest')
            if not noMsrRest: rest.set ('measure', 'yes')
            addElem (nt, rest, lev + 1)
        else:
            p = n.pitch.t           # get pitch elements from parsed tokens
            if len (p) == 3:    acc, step, oct = p
            else:               step, oct = p
            pitch, alter, midi, notehead = s.mkPitch (acc, step, oct, lev + 1)
            if midi: acc = ''       # erase accidental for percussion notes
            addElem (nt, pitch, lev + 1)
        if s.ntup >= 0:                 # modify duration for tuplet notes
            dvs = dvs * s.tmden // s.tmnum
        if dvs:
            addElemT (nt, 'duration', str (dvs), lev + 1)   # skip when dvs == 0, requirement of musicXML
            if not ischord: s.gTime = s.gTime [1], s.gTime [1] + dvs
        ptup = (step, oct)              # pitch tuple without alteration to check for ties
        tstop = ptup in s.ties and s.ties[ptup][2] == s.overlayVnum  # open tie on this pitch tuple in this overlay
        if tstop:
            tie = E.Element ('tie', type='stop')
            addElem (nt, tie, lev + 1)
        if getattr (n, 'tie', 0):
            tie = E.Element ('tie', type='start')
            addElem (nt, tie, lev + 1)
        if (s.midprg != ['', '', '', ''] or midi) and n.name != 'rest': # only add when %%midi was present or percussion
            instId = 'I%s-%s' % (s.pid, 'X' + midi if midi else s.vid)
            chan, midi = ('10', midi) if midi else s.midprg [:2]
            inst = E.Element ('instrument', id=instId)  # instrument id for midi
            addElem (nt, inst, lev + 1)
            if instId not in s.midiInst: s.midiInst [instId] = (s.pid, s.vid, chan, midi, s.midprg [2], s.midprg [3]) # for instrument list in mkScorePart
        addElemT (nt, 'voice', '1', lev + 1)    # default voice, for merging later
        if noMsrRest: addElemT (nt, 'type', xmltype, lev + 1) # add note type if not a measure rest
        for i in range (ndot):          # add dots
            dot = E.Element ('dot')
            addElem (nt, dot, lev + 1)
        decos = s.getNoteDecos (n)      # get decorations for this note
        if acc and not tstop:           # only add accidental if note not tied
            e = E.Element ('accidental')
            if 'courtesy' in decos:
                e.set ('parentheses', 'yes')
                decos.remove ('courtesy')
            e.text = s.accTab [acc]
            addElem (nt, e, lev + 1)
        tupnotation = ''                # start/stop notation element for tuplets
        if s.ntup >= 0:                 # add time modification element for tuplet notes
            tmod = mkTmod (s.tmnum, s.tmden, lev + 1)
            addElem (nt, tmod, lev + 1)
            if s.ntup > 0 and not s.tupnts: tupnotation = 'start'
            s.tupnts.append ((rdvs, tmod))      # remember all tuplet modifiers with corresp. durations
            if s.ntup == 0:             # last tuplet note (and possible chord notes there after)
                if rdvs: tupnotation = 'stop'   # only insert notation in the real note (rdvs > 0)
                s.cmpNormType (rdvs, lev + 1)   # compute and/or add normal-type elements (-> s.ntype)
        hasStem = 1
        if not ischord: s.chordDecos = {}       # clear on non chord note
        if 'stemless' in decos or (s.nostems and n.name != 'rest') or 'stemless' in s.chordDecos:
            hasStem = 0
            addElemT (nt, 'stem', 'none', lev + 1)
            if 'stemless' in decos: decos.remove ('stemless')   # do not handle in doNotations
            if hasattr (n, 'pitches'): s.chordDecos ['stemless'] = 1    # set on first chord note
        if notehead:
            nh = addElemT (nt, 'notehead', re.sub (r'[+-]$', '', notehead), lev + 1)
            if notehead[-1] in '+-': nh.set ('filled', 'yes' if notehead[-1] == '+' else 'no')
        gstaff = s.gStaffNums.get (s.vid, 0)    # staff number of the current voice
        if gstaff: addElemT (nt, 'staff', str (gstaff), lev + 1)
        if hasStem: s.doBeams (n, nt, den, lev + 1)   # no stems -> no beams in a tab staff
        s.doNotations (n, decos, ptup, alter, tupnotation, tstop, nt, lev + 1)
        if n.objs: s.doLyr (n, nt, lev + 1)
        else: s.prevLyric = {}   # clear on note without lyrics
        return nt

    def cmpNormType (s, rdvs, lev): # compute the normal-type of a tuplet (only needed for Finale)
        if rdvs:    # the last real tuplet note (chord notes can still follow afterwards with rdvs == 0)
            durs = [dur for dur, tmod in s.tupnts if dur > 0]
            ndur = sum (durs) // s.tmnum    # duration of the normal type
            s.irrtup = any ((dur != ndur) for dur in durs)  # irregular tuplet
            tix = 16 * s.divisions // ndur  # index in typeMap of normal-type duration
            if tix in s.typeMap:
                s.ntype = str (s.typeMap [tix]) # the normal-type
            else: s.irrtup = 0          # give up, no normal type possible
        if s.irrtup:                    # only add normal-type for irregular tuplets
            for dur, tmod in s.tupnts:  # add normal-type to all modifiers
                addElemT (tmod, 'normal-type', s.ntype, lev + 1)
        s.tupnts = []                   # reset the tuplet buffer

    def doNotations (s, n, decos, ptup, alter, tupnotation, tstop, nt, lev):
        slurs = getattr (n, 'slurs', 0) # slur ends
        pts = getattr (n, 'pitches', [])            # all chord notes available in the first note
        ov = s.overlayVnum                          # current overlay voice number (0 for the main voice)
        if pts:                                     # make list of pitches in chord: [(pitch, octave), ..]
            if type (pts.pitch) == pObj: pts = [pts.pitch]      # chord with one note
            else: pts = [tuple (p.t[-2:]) for p in pts.pitch]   # normal chord
        for pt, (tie_alter, nts, vnum, ntelm) in sorted (list (s.ties.items ())):  # scan all open ties and delete illegal ones
            if vnum != s.overlayVnum: continue      # tie belongs to different overlay
            if pts and pt in pts: continue          # pitch tuple of tie exists in chord
            if getattr (n, 'chord', 0): continue    # skip chord notes
            if pt == ptup: continue                 # skip correct single note tie
            if getattr (n, 'grace', 0): continue    # skip grace notes
            info ('tie between different pitches: %s%s converted to slur' % pt)
            del s.ties [pt]                         # remove the note from pending ties
            e = [t for t in ntelm.findall ('tie') if t.get ('type') == 'start'][0]  # get the tie start element
            ntelm.remove (e)                        # delete start tie element
            e = [t for t in nts.findall ('tied') if t.get ('type') == 'start'][0]   # get the tied start element
            e.tag = 'slur'                          # convert tie into slur
            slurnum = pushSlur (s.slurstack, ov)
            e.set ('number', str (slurnum))
            if slurs: slurs.t.append (')')          # close slur on this note
            else: slurs = pObj ('slurs', [')'])
        tstart = getattr (n, 'tie', 0)  # start a new tie
        if not (tstop or tstart or decos or slurs or s.slurbeg or tupnotation or s.trem): return nt
        nots = E.Element ('notations')  # notation element needed
        if s.trem:  # +/- => tuple tremolo sequence / single note tremolo
            if s.trem < 0: tupnotation = 'single'; s.trem = -s.trem
            if not tupnotation: return  # only add notation at first or last note of a tremolo sequence
            orn = E.Element ('ornaments')
            trm = E.Element ('tremolo', type=tupnotation)   # type = start, stop or single
            trm.text = str (s.trem)     # the number of bars in a tremolo note
            addElem (nots, orn, lev + 1)
            addElem (orn, trm, lev + 2)
            if tupnotation == 'stop' or tupnotation == 'single': s.trem = 0
        elif tupnotation:       # add tuplet type
            tup = E.Element ('tuplet', type=tupnotation)
            if tupnotation == 'start': tup.set ('bracket', 'yes')
            addElem (nots, tup, lev + 1)
        if tstop:               # stop tie
            del s.ties[ptup]    # remove flag
            tie = E.Element ('tied', type='stop')
            addElem (nots, tie, lev + 1)
        if tstart:              # start a tie
            s.ties[ptup] = (alter, nots, s.overlayVnum, nt) # remember pitch tuple to stop tie and apply same alteration
            tie = E.Element ('tied', type='start')
            if tstart.t[0] == '.-': tie.set ('line-type', 'dotted')
            addElem (nots, tie, lev + 1)
        if decos:               # look for slurs and decorations
            slurMap = { '(':1, '.(':1, '(,':1, "('":1, '.(,':1, ".('":1 }
            arts = []           # collect articulations
            for d in decos:     # do all slurs and decos
                if d in slurMap: s.slurbeg.append (d); continue # slurs made in while loop at the end
                elif d == 'fermata' or d == 'H':
                    ntn = E.Element ('fermata', type='upright')
                elif d == 'arpeggio':
                    ntn = E.Element ('arpeggiate', number='1')
                elif d in ['~(', '~)']:
                    if d[1] == '(': tp = 'start'; s.glisnum += 1; gn = s.glisnum
                    else:           tp = 'stop'; gn = s.glisnum; s.glisnum -= 1
                    if s.glisnum < 0: s.glisnum = 0; continue   # stop without previous start
                    ntn = E.Element ('glissando', {'line-type':'wavy', 'number':'%d' % gn, 'type':tp})
                elif d in ['-(', '-)']:
                    if d[1] == '(': tp = 'start'; s.slidenum += 1; gn = s.slidenum
                    else:           tp = 'stop'; gn = s.slidenum; s.slidenum -= 1
                    if s.slidenum < 0: s.slidenum = 0; continue   # stop without previous start
                    ntn = E.Element ('slide', {'line-type':'solid', 'number':'%d' % gn, 'type':tp})
                else: arts.append (d); continue
                addElem (nots, ntn, lev + 1)
            if arts:        # do only note articulations and collect staff annotations in xmldecos
                rest = s.doArticulations (nt, nots, arts, lev + 1)
                if rest: info ('unhandled note decorations: %s' % rest)
        if slurs:           # these are only slur endings
            for d in slurs.t:           # slurs to be closed on this note
                if not s.slurstack.get (ov, 0): break    # no more open old slurs for this (overlay) voice
                slurnum = s.slurstack [ov].pop ()
                slur = E.Element ('slur', number='%d' % slurnum, type='stop')
                addElem (nots, slur, lev + 1)
        while s.slurbeg:    # create slurs beginning on this note
            stp = s.slurbeg.pop (0)
            slurnum = pushSlur (s.slurstack, ov)
            ntn = E.Element ('slur', number='%d' % slurnum, type='start')
            if '.' in stp: ntn.set ('line-type', 'dotted')
            if ',' in stp: ntn.set ('placement', 'below')
            if "'" in stp: ntn.set ('placement', 'above')
            addElem (nots, ntn, lev + 1)            
        if list (nots) != []:    # only add notations if not empty
            addElem (nt, nots, lev)

    def doArticulations (s, nt, nots, arts, lev):
        decos = []
        for a in arts:
            if a in s.artMap:
                art = E.Element ('articulations')
                addElem (nots, art, lev)
                addElem (art, E.Element (s.artMap[a]), lev + 1)
            elif a in s.ornMap:
                orn = E.Element ('ornaments')
                addElem (nots, orn, lev)
                addElem (orn, E.Element (s.ornMap[a]), lev + 1)
            elif a in ['trill(','trill)']:
                orn = E.Element ('ornaments')
                addElem (nots, orn, lev)
                type = 'start' if a.endswith ('(') else 'stop'
                if type == 'start': addElem (orn, E.Element ('trill-mark'), lev + 1)                
                addElem (orn, E.Element ('wavy-line', type=type), lev + 1)                
            elif a in s.tecMap:
                tec = E.Element ('technical')
                addElem (nots, tec, lev)
                addElem (tec, E.Element (s.tecMap[a]), lev + 1)
            elif a in '0123456':
                tec = E.Element ('technical')
                addElem (nots, tec, lev)
                if s.tabStaff == s.pid:             # current voice belongs to a tabStaff
                    alt = int (nt.findtext ('pitch/alter') or 0)    # find midi number of current note
                    step = nt.findtext ('pitch/step')
                    oct = int (nt.findtext ('pitch/octave'))
                    midi = oct * 12 + [0,2,4,5,7,9,11]['CDEFGAB'.index (step)] + alt + 12
                    if a == '0':                    # no string annotation: find one
                        firstFit = ''
                        for smid, istr in s.tunTup: # midi numbers of open strings from high to low
                            if midi >= smid:        # highest open string where this note can be played
                                isvrij = s.strAlloc.isVrij (istr - 1, s.gTime [0], s.gTime [1])
                                a = str (istr)      # string number
                                if not firstFit: firstFit = a
                                if isvrij: break
                        if not isvrij:              # no free string, take the first fit (lowest fret)
                            a = firstFit
                            s.strAlloc.bezet (int (a) - 1, s.gTime [0], s.gTime [1])
                    else:                           # force annotated string number 
                        s.strAlloc.bezet (int (a) - 1, s.gTime [0], s.gTime [1])
                    bmidi = s.tunmid [int (a) - 1]  # midi number of allocated string (with capodastro)
                    fret =  midi - bmidi            # fret position (respecting capodastro)
                    if fret < 25 and fret >= 0:
                        addElemT (tec, 'fret', str (fret), lev + 1)
                    else:
                        altp = 'b' if alt == -1 else '#' if alt == 1 else ''
                        info ('fret %d out of range, note %s%d on string %s' % (fret, step+altp, oct, a))
                    addElemT (tec, 'string', a, lev + 1)
                else:
                    addElemT (tec, 'fingering', a, lev + 1)
            else: decos.append (a)  # return staff annotations
        return decos

    def doLyr (s, n, nt, lev):
        for i, lyrobj in enumerate (n.objs):
            lyrel = E.Element ('lyric', number = str (i + 1))
            if lyrobj.name == 'syl':
                dash = len (lyrobj.t) == 2
                if dash:
                    if i in s.lyrdash:  type = 'middle'
                    else:               type = 'begin'; s.lyrdash [i] = 1
                else:
                    if i in s.lyrdash:  type = 'end';   del s.lyrdash [i]
                    else:               type = 'single'
                addElemT (lyrel, 'syllabic', type, lev + 1)
                txt = lyrobj.t[0]                       # the syllabe
                txt = re.sub (r'(?<!\\)~', ' ', txt)    # replace ~ by space when not escaped (preceded by \)
                txt = re.sub (r'\\(.)', r'\1', txt)     # replace all escaped characters by themselves (for the time being)
                addElemT (lyrel, 'text', txt, lev + 1)
            elif lyrobj.name == 'ext' and i in s.prevLyric:
                pext = s.prevLyric [i].find ('extend')  # identify previous extend
                if pext == None:
                    ext = E.Element ('extend', type = 'start')
                    addElem (s.prevLyric [i], ext, lev + 1)
                elif pext.get('type') == 'stop':        # subsequent extend: stop -> continue
                    pext.set ('type', 'continue')
                ext = E.Element ('extend', type = 'stop')   # always stop on current extend
                addElem (lyrel, ext, lev + 1)
            elif lyrobj.name == 'ext': info ('lyric extend error'); continue
            else: continue          # skip other lyric elements or errors
            addElem (nt, lyrel, lev)
            s.prevLyric [i] = lyrel # for extension (melisma) on the next note

    def doBeams (s, n, nt, den, lev):
        if hasattr (n, 'chord') or hasattr (n, 'grace'):
            s.grcbbrk = s.grcbbrk or n.bbrk.t[0]    # remember if there was any bbrk in or before a grace sequence
            return
        bbrk = s.grcbbrk or n.bbrk.t[0] or den < 32
        s.grcbbrk = False
        if not s.prevNote:  pbm = None
        else:               pbm = s.prevNote.find ('beam')
        bm = E.Element ('beam', number='1')
        bm.text = 'begin'
        if pbm != None:
            if bbrk:
                if pbm.text == 'begin':
                    s.prevNote.remove (pbm)
                elif pbm.text == 'continue':
                    pbm.text = 'end'
                s.prevNote = None
            else: bm.text = 'continue'
        if den >= 32 and n.name != 'rest':
            addElem (nt, bm, lev)
            s.prevNote = nt

    def stopBeams (s):
        if not s.prevNote: return
        pbm = s.prevNote.find ('beam')
        if pbm != None:
            if pbm.text == 'begin':
                s.prevNote.remove (pbm)
            elif pbm.text == 'continue':
                pbm.text = 'end'
        s.prevNote = None

    def staffDecos (s, decos, maat, lev):
        gstaff = s.gStaffNums.get (s.vid, 0)        # staff number of the current voice        
        for d in decos:
            d = s.usrSyms.get (d, d).strip ('!+')   # try to replace user defined symbol
            if d in s.dynaMap:
                dynel = E.Element ('dynamics')
                addDirection (maat, dynel, lev, gstaff, [E.Element (d)], 'below', s.gcue_on)
            elif d in s.wedgeMap:  # wedge
                if ')' in d: type = 'stop'
                else: type = 'crescendo' if '<' in d or 'crescendo' in d else 'diminuendo'
                addDirection (maat, E.Element ('wedge', type=type), lev, gstaff)
            elif d.startswith ('8v'):
                if 'a' in d: type, plce = 'down', 'above'
                else:        type, plce = 'up', 'below'
                if ')' in d: type = 'stop'
                addDirection (maat, E.Element ('octave-shift', type=type, size='8'), lev, gstaff, placement=plce)
            elif d in (['ped','ped-up']):
                type = 'stop' if d.endswith ('up') else 'start'
                addDirection (maat, E.Element ('pedal', type=type), lev, gstaff)
            elif d in ['coda', 'segno']:
                text, attr, val = s.capoMap [d]
                dir = addDirection (maat, E.Element (text), lev, gstaff, placement='above')
                sound = E.Element ('sound'); sound.set (attr, val)
                addElem (dir, sound, lev + 1)
            elif d in s.capoMap:
                text, attr, val = s.capoMap [d]
                words = E.Element ('words'); words.text = text
                dir = addDirection (maat, words, lev, gstaff, placement='above')
                sound = E.Element ('sound'); sound.set (attr, val)
                addElem (dir, sound, lev + 1)
            elif d == '(' or d == '.(': s.slurbeg.append (d)   # start slur on next note
            elif d in ['/-','//-','///-','////-']:  # duplet tremolo sequence
                s.tmnum, s.tmden, s.ntup, s.trem, s.intrem = 2, 1, 2, len (d) - 1, 1
            elif d in ['/','//','///']: s.trem = - len (d)  # single note tremolo
            elif d == 'rbstop': s.rbStop = 1;   # sluit een open volta aan het eind van de maat
            else: s.nextdecos.append (d)    # keep annotation for the next note

    def doFields (s, maat, fieldmap, lev):
        def instDir (midelm, midnum, dirtxt):
            instId = 'I%s-%s' % (s.pid, s.vid)
            words = E.Element ('words'); words.text = dirtxt % midnum
            snd = E.Element ('sound')
            mi = E.Element ('midi-instrument', id=instId)
            dir = addDirection (maat, words, lev, gstaff, placement='above')
            addElem (dir, snd, lev + 1)
            addElem (snd, mi, lev + 2)
            addElemT (mi, midelm, midnum, lev + 3)
        def addTrans (n):
            e = E.Element ('transpose')
            addElemT (e, 'chromatic', n, lev + 2)  # n == signed number string given after transpose
            atts.append ((9, e))
        def doClef (field):
            if re.search (r'perc|map', field):  # percussion clef or new style perc=on or map=perc
                r = re.search (r'(perc|map)\s*=\s*(\S*)', field)
                s.percVoice = 0 if r and r.group (2) not in ['on','true','perc'] else 1
                field = re.sub (r'(perc|map)\s*=\s*(\S*)', '', field)   # erase the perc= for proper clef matching
            clef, gtrans = 0, 0
            clefn = re.search (r'alto1|alto2|alto4|alto|tenor|bass3|bass|treble|perc|none|tab', field)
            clefm = re.search (r"(?:^m=| m=|middle=)([A-Ga-g])([,']*)", field)
            trans_oct2 = re.search (r'octave=([-+]?\d)', field)
            trans = re.search (r'(?:^t=| t=|transpose=)(-?[\d]+)', field)
            trans_oct = re.search (r'([+-^_])(8|15)', field)
            cue_onoff = re.search (r'cue=(on|off)', field)
            strings = re.search (r"strings=(\S+)", field)
            stafflines = re.search (r'stafflines=\s*(\d)', field)
            capo = re.search (r'capo=(\d+)', field)
            if clefn:
                clef = clefn.group ()
            if clefm:
                note, octstr = clefm.groups ()
                nUp = note.upper ()
                octnum = (4 if nUp == note else 5) + (len (octstr) if "'" in octstr else -len (octstr))
                gtrans = (3 if nUp in 'AFD' else 4) - octnum 
                if clef not in ['perc', 'none']: clef = s.clefLineMap [nUp]
            if clef:
                s.gtrans = gtrans   # only change global tranposition when a clef is really defined
                if clef != 'none': s.curClef = clef       # keep track of current abc clef (for percmap)
                sign, line = s.clefMap [clef]
                if not sign: return
                c = E.Element ('clef')
                if gstaff: c.set ('number', str (gstaff))   # only add staff number when defined
                addElemT (c, 'sign', sign, lev + 2)
                if line: addElemT (c, 'line', line, lev + 2)
                if trans_oct:
                    n = trans_oct.group (1) in '-_' and -1 or 1
                    if trans_oct.group (2) == '15': n *= 2  # 8 => 1 octave, 15 => 2 octaves
                    addElemT (c, 'clef-octave-change', str (n), lev + 2) # transpose print out
                    if trans_oct.group (1) in '+-': s.gtrans += n   # also transpose all pitches with one octave
                atts.append ((7, c))
            if trans_oct2:  # octave= can also be in a K: field
                n = int (trans_oct2.group (1))
                s.gtrans = gtrans + n
            if trans != None:   # add transposition in semitones
                e = E.Element ('transpose')
                addElemT (e, 'chromatic', str (trans.group (1)), lev + 3)
                atts.append ((9, e))
            if cue_onoff: s.gcue_on = cue_onoff.group (1) == 'on'
            nlines = 0
            if clef == 'tab':
                s.tabStaff = s.pid
                if capo: s.capo = int (capo.group (1))
                if strings: s.tuning = strings.group (1).split (',')
                s.tunmid = [int (boct) * 12 + [0,2,4,5,7,9,11]['CDEFGAB'.index (bstep)] + 12 + s.capo for bstep, boct in s.tuning]
                s.tunTup = sorted (zip (s.tunmid, range (len (s.tunmid), 0, -1)), reverse=1)
                s.tunmid.reverse ()
                nlines = str (len (s.tuning))
                s.strAlloc.setlines (len (s.tuning), s.pid)
                s.nostems = 'nostems' in field  # tab clef without stems
                s.diafret = 'diafret' in field  # tab with diatonic fretting
            if stafflines or nlines:
                e = E.Element ('staff-details')
                if gstaff: e.set ('number', str (gstaff))   # only add staff number when defined
                if not nlines: nlines = stafflines.group (1)
                addElemT (e, 'staff-lines', nlines, lev + 2)
                if clef == 'tab':
                    for line, t in enumerate (s.tuning):
                        st = E.Element ('staff-tuning', line=str(line+1))
                        addElemT (st, 'tuning-step', t[0], lev + 3)
                        addElemT (st, 'tuning-octave', t[1], lev + 3)
                        addElem (e, st, lev + 2)
                if s.capo: addElemT (e, 'capo', str (s.capo), lev + 2)
                atts.append ((8, e))
        s.diafret = 0           # chromatic fretting is default
        atts = []               # collect xml attribute elements [(order-number, xml-element), ..]
        gstaff = s.gStaffNums.get (s.vid, 0)    # staff number of the current voice
        for ftype, field in fieldmap.items ():
            if not field:       # skip empty fields
                continue
            if ftype == 'Div':  # not an abc field, but handled as if
                d = E.Element ('divisions')
                d.text = field
                atts.append ((1, d))
            elif ftype == 'gstaff':  # make grand staff
                e = E.Element ('staves')
                e.text = str (field)
                atts.append ((4, e))
            elif ftype == 'M':
                if field == 'none': continue
                if field == 'C': field = '4/4'
                elif field == 'C|': field = '2/2'
                t = E.Element ('time')
                if '/' not in field:
                    info ('M:%s not recognized, 4/4 assumed' % field)
                    field = '4/4'
                beats, btype = field.split ('/')[:2]
                try: s.mdur = simplify (eval (beats), int (btype))  # measure duration for Z and X rests (eval allows M:2+3/4)
                except:
                    info ('error in M:%s, 4/4 assumed' % field)
                    s.mdur = (4,4)
                    beats, btype = '4','4'
                addElemT (t, 'beats', beats, lev + 2)
                addElemT (t, 'beat-type', btype, lev + 2)
                atts.append ((3, t))
            elif ftype == 'K':
                accs = ['F','C','G','D','A','E','B']    # == s.sharpness [7:14]
                mode = ''
                key = re.match (r'\s*([A-G][#b]?)\s*([a-zA-Z]*)', field)
                alts = re.search (r'\s((\s?[=^_][A-Ga-g])+)', ' ' + field)  # avoid matching middle=G and m=G
                if key:
                    key, mode = key.groups ()
                    mode = mode.lower ()[:3] # only first three chars, no case
                    if mode not in s.offTab: mode = 'maj'
                    fifths = s.sharpness.index (key) - s.offTab [mode]
                    if fifths >= 0: s.keyAlts = dict (zip (accs[:fifths], fifths * ['1']))
                    else:           s.keyAlts = dict (zip (accs[fifths:], -fifths * ['-1']))
                elif field.startswith ('none') or field == '':  # the default key
                    fifths = 0
                    mode = 'maj'
                if alts:
                    alts = re.findall (r'[=^_][A-Ga-g]', alts.group(1)) # list of explicit alterations
                    alts = [(x[1], s.alterTab [x[0]]) for x in alts]    # [step, alter]
                    for step, alter in alts:                # correct permanent alterations for this key
                        s.keyAlts [step.upper ()] = alter
                    k = E.Element ('key')
                    koctave = []
                    lowerCaseSteps = [step.upper () for step, alter in alts if step.islower ()]
                    for step, alter in sorted (list (s.keyAlts.items ())):
                        if alter == '0':                    # skip neutrals
                            del s.keyAlts [step.upper ()]   # otherwise you get neutral signs on normal notes
                            continue
                        addElemT (k, 'key-step', step.upper (), lev + 2)
                        addElemT (k, 'key-alter', alter, lev + 2)
                        koctave.append ('5' if step in lowerCaseSteps else '4')
                    if koctave:                     # only key signature if not empty
                        for oct in koctave:
                            e = E.Element ('key-octave', number=oct)
                            addElem (k, e, lev + 2)
                        atts.append ((2, k))
                elif mode:
                    k = E.Element ('key')
                    addElemT (k, 'fifths', str (fifths), lev + 2)
                    addElemT (k, 'mode', s.modTab [mode], lev + 2)
                    atts.append ((2, k))
                doClef (field)
            elif ftype == 'L':
                try: s.unitLcur = lmap (int, field.split ('/'))
                except: s.unitLcur = (1,8)
                if len (s.unitLcur) == 1 or s.unitLcur[1] not in s.typeMap:
                    info ('L:%s is not allowed, 1/8 assumed' % field)
                    s.unitLcur = 1,8
            elif ftype == 'V':
                doClef (field)
            elif ftype == 'I':
                s.doField_I (ftype, field, instDir, addTrans)
            elif ftype == 'Q':
                s.doTempo (maat, field, lev)
            elif ftype == 'P':  # ad hoc translation of P: into a staff text direction
                words = E.Element ('rehearsal')
                words.set ('font-weight', 'bold')
                words.text = field
                addDirection (maat, words, lev, gstaff, placement='above')
            elif ftype in 'TCOAZNGHRBDFSU':
                info ('**illegal header field in body: %s, content: %s' % (ftype, field))
            else:
                info ('unhandled field: %s, content: %s' % (ftype, field))

        if atts:
            att = E.Element ('attributes')      # insert sub elements in the order required by musicXML
            addElem (maat, att, lev)
            for _, att_elem in sorted (atts, key=lambda x: x[0]):   # ordering !
                addElem (att, att_elem, lev + 1)
        if s.diafret:
            other = E.Element ('other-direction'); other.text = 'diatonic fretting'
            addDirection (maat, other, lev, 0)

    def doTempo (s, maat, field, lev):
        gstaff = s.gStaffNums.get (s.vid, 0)    # staff number of the current voice
        t = re.search (r'(\d)/(\d\d?)\s*=\s*(\d[.\d]*)|(\d[.\d]*)', field)
        rtxt = re.search (r'"([^"]*)"', field) # look for text in Q: field
        if not t and not rtxt: return
        elems = []  # [(element, sub-elements)] will be added as direction-types
        if rtxt:
            num, den, upm = 1, 4, s.tempoMap.get (rtxt.group (1).lower ().strip (), 120)
            words = E.Element ('words'); words.text = rtxt.group (1)
            elems.append ((words, []))
        if t:
            try:
                if t.group (4): num, den, upm = 1, s.unitLcur[1] , float (t.group (4))  # old syntax Q:120
                else:           num, den, upm = int (t.group (1)), int (t.group (2)), float (t.group (3))
            except: info ('conversion error: %s' % field); return
            num, den = simplify (num, den);
            dotted, den_not = (1, den // 2) if num == 3 else (0, den)
            metro = E.Element ('metronome')
            u = E.Element ('beat-unit'); u.text = s.typeMap.get (4 * den_not, 'quarter')
            pm = E.Element ('per-minute'); pm.text = ('%.2f' % upm).rstrip ('0').rstrip ('.')
            subelms = [u, E.Element ('beat-unit-dot'), pm] if dotted else [u, pm]
            elems.append ((metro, subelms))
        dir = addDirection (maat, elems, lev, gstaff, [], placement='above')
        if num != 1 and num != 3: info ('in Q: numerator in %d/%d not supported' % (num, den))
        qpm = 4. * num * upm / den
        sound = E.Element ('sound'); sound.set ('tempo', '%.2f' % qpm)
        addElem (dir, sound, lev + 1)

    def mkBarline (s, maat, loc, lev, style='', dir='', ending=''):
        b = E.Element ('barline', location=loc)
        if style:
            addElemT (b, 'bar-style', style, lev + 1)
        if s.curVolta:    # first stop a current volta
            end = E.Element ('ending', number=s.curVolta, type='stop')
            s.curVolta = ''
            if loc == 'left':   # stop should always go to a right barline
                bp = E.Element ('barline', location='right')
                addElem (bp, end, lev + 1)
                addElem (s.prevmsre, bp, lev)   # prevmsre has no right barline! (ending would have stopped there)
            else:
                addElem (b, end, lev + 1)
        if ending:
            ending = ending.replace ('-',',')   # MusicXML only accepts comma's
            endtxt = ''
            if ending.startswith ('"'):     # ending is a quoted string
                endtxt = ending.strip ('"')
                ending = '33'               # any number that is not likely to occur elsewhere
            end = E.Element ('ending', number=ending, type='start')
            if endtxt: end.text = endtxt    # text appears in score in stead of number attribute
            addElem (b, end, lev + 1)
            s.curVolta = ending
        if dir:
            r = E.Element ('repeat', direction=dir)
            addElem (b, r, lev + 1)
        addElem (maat, b, lev)

    def doChordSym (s, maat, sym, lev):
        alterMap = {'#':'1','=':'0','b':'-1'}
        rnt = sym.root.t
        chord = E.Element ('harmony')
        addElem (maat, chord, lev)
        root = E.Element ('root')
        addElem (chord, root, lev + 1)
        addElemT (root, 'root-step', rnt[0], lev + 2)
        if len (rnt) == 2: addElemT (root, 'root-alter', alterMap [rnt[1]], lev + 2)
        kind = s.chordTab.get (sym.kind.t[0], 'major') if sym.kind.t else 'major'
        addElemT (chord, 'kind', kind, lev + 1)
        if hasattr (sym, 'bass'):
            bnt = sym.bass.t
            bass = E.Element ('bass')
            addElem (chord, bass, lev + 1)
            addElemT (bass, 'bass-step', bnt[0], lev + 2)
            if len (bnt) == 2: addElemT (bass, 'bass-alter', alterMap [bnt[1]], lev + 2)
        degs = getattr (sym, 'degree', '')
        if degs:
            if type (degs) != list_type: degs = [degs]
            for deg in degs:
                deg = deg.t[0]
                if deg[0] == '#':   alter = '1';  deg = deg[1:]
                elif deg[0] == 'b': alter = '-1'; deg = deg[1:]
                else:               alter = '0';  deg = deg
                degree = E.Element ('degree')
                addElem (chord, degree, lev + 1)
                addElemT (degree, 'degree-value', deg, lev + 2)
                addElemT (degree, 'degree-alter', alter, lev + 2)
                addElemT (degree, 'degree-type', 'add', lev + 2)

    def mkMeasure (s, i, t, lev, fieldmap={}):
        s.msreAlts = {}
        s.ntup, s.trem, s.intrem = -1, 0, 0
        s.acciatura = 0 # next grace element gets acciatura attribute
        s.rbStop = 0    # sluit een open volta aan het eind van de maat
        overlay = 0
        maat = E.Element ('measure', number = str(i))
        if fieldmap: s.doFields (maat, fieldmap, lev + 1)
        if s.linebrk:   # there was a line break in the previous measure
            e = E.Element ('print')
            e.set ('new-system', 'yes')
            addElem (maat, e, lev + 1)
            s.linebrk = 0
        for it, x in enumerate (t):
            if x.name == 'note' or x.name == 'rest':
                if x.dur.t[0] == 0:  # a leading zero was used for stemmless in abcm2ps, we only support !stemless!
                    x.dur.t = tuple ([1, x.dur.t[1]])
                note = s.mkNote (x, lev + 1)
                addElem (maat, note, lev + 1)
            elif x.name == 'lbar':
                bar = x.t[0]
                if bar == '|' or bar == '[|': pass # skip redundant bar
                elif ':' in bar:    # forward repeat
                    volta = x.t[1] if len (x.t) == 2  else ''
                    s.mkBarline (maat, 'left', lev + 1, style='heavy-light', dir='forward', ending=volta)
                else:               # bar must be a volta number
                    s.mkBarline (maat, 'left', lev + 1, ending=bar)
            elif x.name == 'rbar':
                bar = x.t[0]
                if bar == '.|':
                    s.mkBarline (maat, 'right', lev + 1, style='dotted')
                elif ':' in bar:  # backward repeat
                    s.mkBarline (maat, 'right', lev + 1, style='light-heavy', dir='backward')
                elif bar == '||':
                    s.mkBarline (maat, 'right', lev + 1, style='light-light')
                elif bar == '[|]' or bar == '[]':
                    s.mkBarline (maat, 'right', lev + 1, style='none')
                elif '[' in bar or ']' in bar:
                    s.mkBarline (maat, 'right', lev + 1, style='light-heavy')
                elif bar == '|' and s.rbStop:   # normale barline hoeft niet, behalve om een volta te stoppen
                    s.mkBarline (maat, 'right', lev + 1, style='regular')
                elif bar[0] == '&': overlay = 1
            elif x.name == 'tup':
                if   len (x.t) == 3: n, into, nts = x.t
                elif len (x.t) == 2: n, into, nts = x.t + [0]
                else:                n, into, nts = x.t[0], 0, 0
                if into == 0: into = 3 if n in [2,4,8] else 2
                if nts == 0: nts = n
                s.tmnum, s.tmden, s.ntup = n, into, nts
            elif x.name == 'deco':
                s.staffDecos (x.t, maat, lev + 1)   # output staff decos, postpone note decos to next note
            elif x.name == 'text':
                pos, text = x.t[:2]
                place = 'above' if pos == '^' else 'below'
                words = E.Element ('words')
                words.text = text
                gstaff = s.gStaffNums.get (s.vid, 0)    # staff number of the current voice
                addDirection (maat, words, lev + 1, gstaff, placement=place)
            elif x.name == 'inline':
                fieldtype, fieldval = x.t[0], ' '.join (x.t[1:])
                s.doFields (maat, {fieldtype:fieldval}, lev + 1)
            elif x.name == 'accia': s.acciatura = 1
            elif x.name == 'linebrk':
                s.supports_tag = 1
                if it > 0 and t[it -1].name == 'lbar':  # we are at start of measure
                    e = E.Element ('print')             # output linebreak now
                    e.set ('new-system', 'yes')
                    addElem (maat, e, lev + 1)
                else:
                    s.linebrk = 1   # output linebreak at start of next measure
            elif x.name == 'chordsym':
                s.doChordSym (maat, x, lev + 1)
        s.stopBeams ()
        s.prevmsre = maat
        return maat, overlay

    def mkPart (s, maten, id, lev, attrs, nstaves, rOpt):
        s.slurstack = {}
        s.glisnum = 0;          # xml number attribute for glissandos
        s.slidenum = 0;         # xml number attribute for slides
        s.unitLcur = s.unitL    # set the default unit length at begin of each voice
        s.curVolta = ''
        s.lyrdash = {}
        s.linebrk = 0
        s.midprg = ['', '', '', ''] # MIDI channel nr, program nr, volume, panning for the current part
        s.gcue_on = 0           # reset cue note marker for each new voice
        s.gtrans = 0            # reset octave transposition (by clef)
        s.percVoice = 0         # 1 if percussion clef encountered
        s.curClef = ''          # current abc clef (for percmap)
        s.nostems = 0           # for the tab clef
        s.tuning = s.tuningDef  # reset string tuning to default
        part = E.Element ('part', id=id)
        s.overlayVnum = 0       # overlay voice number to relate ties that extend from one overlayed measure to the next
        gstaff = s.gStaffNums.get (s.vid, 0)    # staff number of the current voice
        attrs_cpy = attrs.copy ()   # don't change attrs itself in next line
        if gstaff == 1: attrs_cpy ['gstaff'] = nstaves  # make a grand staff
        if 'perc' in attrs_cpy.get ('V', ''): del attrs_cpy ['K'] # remove key from percussion voice
        msre, overlay = s.mkMeasure (1, maten[0], lev + 1, attrs_cpy)
        addElem (part, msre, lev + 1)
        for i, maat in enumerate (maten[1:]):
            s.overlayVnum = s.overlayVnum + 1 if overlay else 0
            msre, next_overlay = s.mkMeasure (i+2, maat, lev + 1)
            if overlay: mergePartMeasure (part, msre, s.overlayVnum, rOpt)
            else:       addElem (part, msre, lev + 1)
            overlay = next_overlay
        return part

    def mkScorePart (s, id, vids_p, partAttr, lev):
        def mkInst (instId, vid, midchan, midprog, midnot, vol, pan, lev):
            si = E.Element ('score-instrument', id=instId)
            pnm = partAttr.get (vid, [''])[0]   # part name if present
            addElemT (si, 'instrument-name', pnm or 'dummy', lev + 2)   # MuseScore needs a name
            mi = E.Element ('midi-instrument', id=instId)
            if midchan: addElemT (mi, 'midi-channel', midchan, lev + 2)
            if midprog: addElemT (mi, 'midi-program', str (int (midprog) + 1), lev + 2) # compatible with abc2midi
            if midnot:  addElemT (mi, 'midi-unpitched', str (int (midnot) + 1), lev + 2)
            if vol: addElemT (mi, 'volume', '%.2f' % (int (vol) / 1.27), lev + 2)
            if pan: addElemT (mi, 'pan', '%.2f' % (int (pan) / 127. * 180 - 90), lev + 2)
            return (si, mi)
        naam, subnm, midprg = partAttr [id]
        sp = E.Element ('score-part', id='P'+id)
        nm = E.Element ('part-name')
        nm.text = naam
        addElem (sp, nm, lev + 1)
        snm = E.Element ('part-abbreviation')
        snm.text = subnm
        if subnm: addElem (sp, snm, lev + 1)    # only add if subname was given
        inst = []
        for instId, (pid, vid, chan, midprg, vol, pan) in sorted (s.midiInst.items ()):
            midprg, midnot = ('0', midprg) if chan == '10' else (midprg, '')
            if pid == id: inst.append (mkInst (instId, vid, chan, midprg, midnot, vol, pan, lev))
        for si, mi in inst: addElem (sp, si, lev + 1)
        for si, mi in inst: addElem (sp, mi, lev + 1)
        return sp

    def mkPartlist (s, vids, partAttr, lev):
        def addPartGroup (sym, num):
            pg = E.Element ('part-group', number=str (num), type='start')
            addElem (partlist, pg, lev + 1)
            addElemT (pg, 'group-symbol', sym, lev + 2)
            addElemT (pg, 'group-barline', 'yes', lev + 2)
        partlist = E.Element ('part-list')
        g_num = 0       # xml group number
        for g in (s.groups or vids):    # brace/bracket or abc_voice_id
            if   g == '[': g_num += 1; addPartGroup ('bracket', g_num)
            elif g == '{': g_num += 1; addPartGroup ('brace', g_num)
            elif g in '}]':
                pg = E.Element ('part-group', number=str (g_num), type='stop')
                addElem (partlist, pg, lev + 1)
                g_num -= 1
            else:   # g = abc_voice_id
                if g not in vids: continue  # error in %%score
                sp = s.mkScorePart (g, vids, partAttr, lev + 1)
                addElem (partlist, sp, lev + 1)
        return partlist

    def doField_I (s, type, x, instDir, addTrans):
        def instChange (midchan, midprog):  # instDir -> doFields
            if midchan and midchan != s.midprg [0]: instDir ('midi-channel', midchan, 'chan: %s')
            if midprog and midprog != s.midprg [1]: instDir ('midi-program', str (int (midprog) + 1), 'prog: %s')
        def readPfmt (x, n): # read ABC page formatting constant
            if not s.pageFmtAbc: s.pageFmtAbc = s.pageFmtDef    # set the default values on first change
            ro = re.search (r'[^.\d]*([\d.]+)\s*(cm|in|pt)?', x)    # float followed by unit
            if ro:
                x, unit = ro.groups ()  # unit == None when not present
                u = {'cm':10., 'in':25.4, 'pt':25.4/72} [unit] if unit else 1.
                s.pageFmtAbc [n] = float (x) * u   # convert ABC values to millimeters
            else: info ('error in page format: %s' % x)
        def readPercMap (x):    # parse I:percmap <abc_note> <step> <MIDI> <notehead>
            def getMidNum (sndnm):          # find midi number of GM drum sound name
                pnms = sndnm.split ('-')    # sound name parts (from I:percmap)
                ps = s.percsnd [:]          # copy of the instruments
                _f = lambda ip, xs, pnm: ip < len (xs) and xs[ip].find (pnm) > -1   # part xs[ip] and pnm match
                for ip, pnm in enumerate (pnms):    # match all percmap sound name parts
                    ps = [(nm, mnum) for nm, mnum in ps if _f (ip, nm.split ('-'), pnm) ]   # filter instruments
                    if len (ps) <= 1: break # no match or one instrument left
                if len (ps) == 0: info ('drum sound: %s not found' % sndnm); return '38'
                return ps [0][1]            # midi number of (first) instrument found
            def midiVal (acc, step, oct):   # abc note -> midi note number
                oct = (4 if step.upper() == step else 5) + int (oct)
                return oct * 12 + [0,2,4,5,7,9,11]['CDEFGAB'.index (step.upper())] + {'^':1,'_':-1,'=':0}.get (acc, 0) + 12
            p0, p1, p2, p3, p4 = abc_percmap.parseString (x).asList ()  # percmap, abc-note, display-step, midi, note-head
            acc, astep, aoct = p1
            nstep, noct = (astep, aoct) if p2 == '*' else p2
            if p3 == '*':                           midi = str (midiVal (acc, astep, aoct))
            elif isinstance (p3, list_type):        midi = str (midiVal (p3[0], p3[1], p3[2]))
            elif isinstance (p3, int_type):         midi = str (p3)
            else:                                   midi = getMidNum (p3.lower ())
            head = re.sub (r'(.)-([^x])', r'\1 \2', p4) # convert abc note head names to xml
            s.percMap [(s.pid, acc + astep, aoct)] = (nstep, noct, midi, head)
        if x.startswith ('score') or x.startswith ('staves'):
            s.staveDefs += [x]          # collect all voice mappings
        elif x.startswith ('staffwidth'): info ('skipped I-field: %s' % x)
        elif x.startswith ('staff'):    # set new staff number of the current voice
            r1 = re.search (r'staff *([+-]?)(\d)', x)
            if r1:
                sign = r1.group (1)
                num = int (r1.group (2))
                gstaff = s.gStaffNums.get (s.vid, 0)    # staff number of the current voice
                if sign:                                # relative staff number
                    num = (sign == '-') and gstaff - num or gstaff + num
                else:                                   # absolute abc staff number
                    try: vabc = s.staves [num - 1][0]   # vid of (first voice of) abc-staff num
                    except: vabc = 0; info ('abc staff %s does not exist' % num)
                    num = s.gStaffNumsOrg.get (vabc, 0) # xml staff number of abc-staff num
                if gstaff and num > 0 and num <= s.gNstaves [s.vid]:
                    s.gStaffNums [s.vid] = num
                else: info ('could not relocate to staff: %s' % r1.group ())
            else: info ('not a valid staff redirection: %s' % x)
        elif x.startswith ('scale'): readPfmt (x, 0)
        elif x.startswith ('pageheight'): readPfmt (x, 1)
        elif x.startswith ('pagewidth'): readPfmt (x, 2)
        elif x.startswith ('leftmargin'): readPfmt (x, 3)
        elif x.startswith ('rightmargin'): readPfmt (x, 4)
        elif x.startswith ('topmargin'): readPfmt (x, 5)
        elif x.startswith ('botmargin'): readPfmt (x, 6)
        elif x.startswith ('MIDI') or x.startswith ('midi'):
            r1 = re.search (r'program *(\d*) +(\d+)', x)
            r2 = re.search (r'channel *(\d+)', x)
            r3 = re.search (r"drummap\s+([_=^]*)([A-Ga-g])([,']*)\s+(\d+)", x)
            r4 = re.search (r'control *(\d+) +(\d+)', x)
            ch_nw, prg_nw, vol_nw, pan_nw = '', '', '', ''
            if r1: ch_nw, prg_nw = r1.groups () # channel nr or '', program nr
            if r2: ch_nw = r2.group (1)         # channel nr only
            if r4:
                cnum, cval = r4.groups ()       # controller number, controller value
                if cnum == '7': vol_nw = cval
                if cnum == '10': pan_nw = cval
            if r1 or r2 or r4:
                ch  = ch_nw  or s.midprg [0]
                prg = prg_nw or s.midprg [1]
                vol = vol_nw or s.midprg [2]
                pan = pan_nw or s.midprg [3]
                instId = 'I%s-%s' % (s.pid, s.vid)              # only look for real instruments, no percussion
                if instId in s.midiInst: instChange (ch, prg)   # instChance -> doFields
                s.midprg = [ch, prg, vol, pan]  # mknote: new instrument -> s.midiInst
            if r3:      # translate drummap to percmap
                acc, step, oct, midi = r3.groups ()
                oct = -len (oct) if ',' in x else len (oct)
                notehead = 'x' if acc == '^' else 'circle-x' if acc == '_' else 'normal'
                s.percMap [(s.pid, acc + step, oct)] = (step, oct, midi, notehead)
            r = re.search (r'transpose[^-\d]*(-?\d+)', x)
            if r: addTrans (r.group (1))        # addTrans -> doFields
        elif x.startswith ('percmap'): readPercMap (x); s.pMapFound = 1
        else: info ('skipped I-field: %s' % x)

    def parseStaveDef (s, vdefs):
        for vid in vdefs: s.vcepid [vid] = vid              # default: each voice becomes an xml part
        if not s.staveDefs: return vdefs
        for x in s.staveDefs [1:]: info ('%%%%%s dropped, multiple stave mappings not supported' % x)
        x = s.staveDefs [0]                                 # only the first %%score is honoured
        score = abc_scoredef.parseString (x) [0]
        f = lambda x: type (x) == uni_type and [x] or x
        s.staves = lmap (f, mkStaves (score, vdefs))        # [[vid] for each staff]
        s.grands = lmap (f, mkGrand (score, vdefs))         # [staff-id], staff-id == [vid][0]
        s.groups = mkGroups (score)
        vce_groups = [vids for vids in s.staves if len (vids) > 1]  # all voice groups
        d = {}                                              # for each voice group: map first voice id -> all merged voice ids
        for vgr in vce_groups: d [vgr[0]] = vgr
        for gstaff in s.grands:                             # for all grand staves
            if len (gstaff) == 1: continue                  # skip single parts
            for v, stf_num in zip (gstaff, range (1, len (gstaff) + 1)):
                for vx in d.get (v, [v]):                   # allocate staff numbers
                    s.gStaffNums [vx] = stf_num             # to all constituant voices
                    s.gNstaves [vx] = len (gstaff)          # also remember total number of staves
        s.gStaffNumsOrg = s.gStaffNums.copy ()              # keep original allocation for abc -> xml staff map
        for xmlpart in s.grands:
            pid = xmlpart [0]                               # part id == first staff id == first voice id
            vces = [v for stf in xmlpart for v in d.get (stf, [stf])]
            for v in vces: s.vcepid [v] = pid
        return vdefs

    def voiceNamesAndMaps (s, ps):  # get voice names and mappings
        vdefs = {}
        for vid, vcedef, vce in ps: # vcedef == emtpy or first pObj == voice definition
            pname, psubnm = '', ''  # part name and abbreviation
            if not vcedef:          # simple abc without voice definitions
                vdefs [vid] =  pname, psubnm, ''
            else:                   # abc with voice definitions
                if vid != vcedef.t[1]: info ('voice ids unequal: %s (reg-ex) != %s (grammar)' % (vid, vcedef.t[1]))
                rn = re.search (r'(?:name|nm)="([^"]*)"', vcedef.t[2])
                if rn: pname = rn.group (1)
                rn = re.search (r'(?:subname|snm|sname)="([^"]*)"', vcedef.t[2])
                if rn: psubnm = rn.group (1)
                vcedef.t[2] = vcedef.t[2].replace ('"%s"' % pname, '""').replace ('"%s"' % psubnm, '""')   # clear voice name to avoid false clef matches later on
                vdefs [vid] =  pname, psubnm, vcedef.t[2]
            xs = [pObj.t[1] for maat in vce for pObj in maat if pObj.name == 'inline']  # all inline statements in vce
            s.staveDefs += [x.replace ('%5d',']') for x in xs if x.startswith ('score') or x.startswith ('staves')] # filter %%score and %%staves
        return vdefs

    def doHeaderField (s, fld, attrmap):
        type, value = fld.t[0], fld.t[1].replace ('%5d',']')    # restore closing brackets (see splitHeaderVoices)
        if not value:    # skip empty field
            return
        if type == 'M':
            attrmap [type] = value
        elif type == 'L':
            try: s.unitL = lmap (int, fld.t[1].split ('/'))
            except:
                info ('illegal unit length:%s, 1/8 assumed' % fld.t[1])
                s.unitL = 1,8
            if len (s.unitL) == 1 or s.unitL[1] not in s.typeMap:
                info ('L:%s is not allowed, 1/8 assumed' % fld.t[1])
                s.unitL = 1,8
        elif type == 'K':
            attrmap[type] = value
        elif type == 'T':
            s.title = s.title + '\n' + value if s.title else value
        elif type == 'U':
            sym = fld.t[2].strip ('!+')
            s.usrSyms [value] = sym
        elif type == 'I':
            s.doField_I (type, value, lambda x,y,z:0, lambda x:0)
        elif type == 'Q':
            attrmap[type] = value
        elif type in 'CRZNOAGHBDFSP':           # part maps are treated as meta data
            type = s.metaMap.get (type, type)   # respect the (user defined --meta) mapping of various ABC fields to XML meta data types
            c = s.metadata.get (type, '')
            s.metadata [type] = c + '\n' + value if c else value    # concatenate multiple info fields with new line as separator
        else:
            info ('skipped header: %s' % fld)

    def mkIdentification (s, score, lev):
        if s.title:
            xs = s.title.split ('\n')   # the first T: line goes to work-title
            ys = '\n'.join (xs [1:])    # place subsequent T: lines into work-number
            w = E.Element ('work')
            addElem (score, w, lev + 1)
            if ys: addElemT (w, 'work-number', ys, lev + 2)
            addElemT (w, 'work-title', xs[0], lev + 2)
        ident = E.Element ('identification')
        addElem (score, ident, lev + 1)
        for mtype, mval in s.metadata.items ():
            if mtype in s.metaTypes and mtype != 'rights':    # all metaTypes are MusicXML creator types
                c = E.Element ('creator', type=mtype)
                c.text = mval
                addElem (ident, c, lev + 2)
        if 'rights' in s.metadata:
            c = addElemT (ident, 'rights', s.metadata ['rights'], lev + 2)
        encoding = E.Element ('encoding')
        addElem (ident, encoding, lev + 2)
        encoder = E.Element ('encoder')
        encoder.text = 'abc2xml version %d' % VERSION
        addElem (encoding, encoder, lev + 3)
        if s.supports_tag:  # avoids interference of auto-flowing and explicit linebreaks
            suports = E.Element ('supports', attribute="new-system", element="print", type="yes", value="yes")
            addElem (encoding, suports, lev + 3)
        encodingDate = E.Element ('encoding-date')
        encodingDate.text = str (datetime.date.today ())
        addElem (encoding, encodingDate, lev + 3)
        s.addMeta (ident, lev + 2)

    def mkDefaults (s, score, lev):
        if s.pageFmtCmd: s.pageFmtAbc = s.pageFmtCmd
        if not s.pageFmtAbc: return # do not output the defaults if none is desired
        abcScale, h, w, l, r, t, b = s.pageFmtAbc
        space = abcScale * 2.117    # 2.117 = 6pt = space between staff lines for scale = 1.0 in abcm2ps
        mils = 4 * space    # staff height in millimeters
        scale = 40. / mils  # tenth's per millimeter
        dflts = E.Element ('defaults')
        addElem (score, dflts, lev)
        scaling = E.Element ('scaling')
        addElem (dflts, scaling, lev + 1)
        addElemT (scaling, 'millimeters', '%g' % mils, lev + 2)
        addElemT (scaling, 'tenths', '40', lev + 2)
        layout = E.Element ('page-layout')
        addElem (dflts, layout, lev + 1)
        addElemT (layout, 'page-height', '%g' % (h * scale), lev + 2)
        addElemT (layout, 'page-width', '%g' % (w * scale), lev + 2)
        margins = E.Element ('page-margins', type='both')
        addElem (layout, margins, lev + 2)
        addElemT (margins, 'left-margin', '%g' % (l * scale), lev + 3)
        addElemT (margins, 'right-margin', '%g' % (r * scale), lev + 3)
        addElemT (margins, 'top-margin', '%g' % (t * scale), lev + 3)
        addElemT (margins, 'bottom-margin', '%g' % (b * scale), lev + 3)

    def addMeta (s, parent, lev):
        misc = E.Element ('miscellaneous')
        mf = 0
        for mtype, mval in sorted (s.metadata.items ()):
            if mtype == 'S':
                addElemT (parent, 'source', mval, lev)
            elif mtype in s.metaTypes: continue  # mapped meta data has already been output (in creator elements)
            else:
                mf = E.Element ('miscellaneous-field', name=s.metaTab [mtype])
                mf.text = mval
                addElem (misc, mf, lev + 1)
        if mf != 0: addElem (parent, misc, lev)

    def parse (s, abc_string, rOpt=False, bOpt=False, fOpt=False):
        abctext = abc_string.replace ('[I:staff ','[I:staff')  # avoid false beam breaks
        s.reset (fOpt)
        header, voices = splitHeaderVoices (abctext)
        ps = []
        try:
            lbrk_insert = 0 if re.search (r'I:linebreak\s*([!$]|none)|I:continueall\s*(1|true)', header) else bOpt
            hs = abc_header.parseString (header) if header else ''
            for id, voice in voices:
                if lbrk_insert:                                 # insert linebreak at EOL
                    r1 = re.compile (r'\[[wA-Z]:[^]]*\]')       # inline field
                    has_abc = lambda x: r1.sub ('', x).strip () # empty if line only contains inline fields
                    voice = '\n'.join ([balk.rstrip ('$!') + '$' if has_abc (balk) else balk for balk in voice.splitlines ()])
                prevLeftBar = None      # previous voice ended with a left-bar symbol (double repeat)
                s.orderChords = s.fOpt and ('tab' in voice [:200] or [x for x in hs if x.t[0] == 'K' and 'tab' in x.t[1]])
                vce = abc_voice.parseString (voice).asList ()
                lyr_notes = []          # remember notes between lyric blocks
                for m in vce:           # all measures
                    for e in m:         # all abc-elements
                        if e.name == 'lyr_blk':         # -> e.objs is list of lyric lines
                            lyr = [line.objs for line in e.objs]    # line.objs is listof syllables
                            alignLyr (lyr_notes, lyr)   # put all syllables into corresponding notes
                            lyr_notes = []
                        else:
                            lyr_notes.append (e)
                if not vce:             # empty voice, insert an inline field that will be rejected
                    vce = [[pObj ('inline', ['I', 'empty voice'])]]
                if prevLeftBar:
                    vce[0].insert (0, prevLeftBar)  # insert at begin of first measure
                    prevLeftBar = None
                if vce[-1] and vce[-1][-1].name == 'lbar':  # last measure ends with an lbar
                    prevLeftBar = vce[-1][-1]
                    if len (vce) > 1:   # vce should not become empty (-> exception when taking vcelyr [0][0])
                        del vce[-1]     # lbar was the only element in measure vce[-1]
                vcelyr = vce
                elem1 = vcelyr [0][0]   # the first element of the first measure
                if  elem1.name == 'inline'and elem1.t[0] == 'V':    # is a voice definition
                    voicedef = elem1 
                    del vcelyr [0][0]   # do not read voicedef twice
                else:
                    voicedef = ''
                ps.append ((id, voicedef, vcelyr))
        except ParseException as err:
            if err.loc > 40:    # limit length of error message, compatible with markInputline
                err.pstr = err.pstr [err.loc - 40: err.loc + 40]
                err.loc = 40
            xs = err.line[err.col-1:]
            info (err.line, warn=0)
            info ((err.col-1) * '-' + '^', warn=0)
            if   re.search (r'\[U:', xs):
                info ('Error: illegal user defined symbol: %s' % xs[1:], warn=0)
            elif re.search (r'\[[OAPZNGHRBDFSXTCIU]:', xs):
                info ('Error: header-only field %s appears after K:' % xs[1:], warn=0)
            else:
                info ('Syntax error at column %d' % err.col, warn=0)
            raise

        score = E.Element ('score-partwise')
        attrmap = {'Div': str (s.divisions), 'K':'C treble', 'M':'4/4'}
        for res in hs:
            if res.name == 'field':
                s.doHeaderField (res, attrmap)
            else:
                info ('unexpected header item: %s' % res)

        vdefs = s.voiceNamesAndMaps (ps)
        vdefs = s.parseStaveDef (vdefs)

        lev = 0
        vids, parts, partAttr = [], [], {}
        s.strAlloc = stringAlloc ()
        for vid, _, vce in ps:          # voice id, voice parse tree
            pname, psubnm, voicedef = vdefs [vid]   # part name
            attrmap ['V'] = voicedef    # abc text of first voice definition (after V:vid) or empty
            pid = 'P%s' % vid           # let part id start with an alpha
            s.vid = vid                 # avoid parameter passing, needed in mkNote for instrument id
            s.pid = s.vcepid [s.vid]    # xml part-id for the current voice
            s.gTime = (0, 0)            # reset time
            s.strAlloc.beginZoek ()     # reset search index
            part = s.mkPart (vce, pid, lev + 1, attrmap, s.gNstaves.get (vid, 0), rOpt)
            if 'Q' in attrmap: del attrmap ['Q']    # header tempo only in first part
            parts.append (part)
            vids.append (vid)
            partAttr [vid] = (pname, psubnm, s.midprg)
            if s.midprg != ['', '', '', ''] and not s.percVoice:    # when a part has only rests
                instId = 'I%s-%s' % (s.pid, s.vid)
                if instId not in s.midiInst: s.midiInst [instId] = (s.pid, s.vid, s.midprg [0], s.midprg [1], s.midprg [2], s.midprg [3])
        parts, vidsnew = mergeParts (parts, vids, s.staves, rOpt) # merge parts into staves as indicated by %%score
        parts, vidsnew = mergeParts (parts, vidsnew, s.grands, rOpt, 1) # merge grand staves
        reduceMids (parts, vidsnew, s.midiInst)

        s.mkIdentification (score, lev)
        s.mkDefaults (score, lev + 1)

        partlist = s.mkPartlist (vids, partAttr, lev + 1)
        addElem (score, partlist, lev + 1)
        for ip, part in enumerate (parts): addElem (score, part, lev + 1)

        return score


def decodeInput (data_string):
    try:        enc = 'utf-8';   unicode_string = data_string.decode (enc)
    except:
        try:    enc = 'latin-1'; unicode_string = data_string.decode (enc)
        except: raise ValueError ('data not encoded in utf-8 nor in latin-1')
    info ('decoded from %s' % enc)
    return unicode_string

def ggd (a, b): # greatest common divisor
    return a if b == 0 else ggd (b, a % b)

xmlVersion = "<?xml version='1.0' encoding='utf-8'?>"    
def fixDoctype (elem):
    if python3: xs = E.tostring (elem, encoding='unicode')  # writing to file will auto-encode to utf-8
    else:       xs = E.tostring (elem, encoding='utf-8')    # keep the string utf-8 encoded for writing to file
    ys = xs.split ('\n')
    ys.insert (0, xmlVersion)  # crooked logic of ElementTree lib
    ys.insert (1, '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">')
    return '\n'.join (ys)

def xml2mxl (pad, fnm, data):   # write xml data to compressed .mxl file
    from zipfile import ZipFile, ZIP_DEFLATED
    fnmext = fnm + '.xml'       # file name with extension, relative to the root within the archive
    outfile = os.path.join (pad, fnm + '.mxl')
    meta  = '%s\n<container><rootfiles>\n' % xmlVersion
    meta += '<rootfile full-path="%s" media-type="application/vnd.recordare.musicxml+xml"/>\n' % fnmext
    meta += '</rootfiles></container>'
    f = ZipFile (outfile, 'w', ZIP_DEFLATED)
    f.writestr ('META-INF/container.xml', meta)
    f.writestr (fnmext, data)
    f.close ()
    info ('%s written' % outfile, warn=0)

def convert (pad, fnm, abc_string, mxl, rOpt=False, tOpt=False, bOpt=False, fOpt=False):  # not used, backwards compatibility
    score = mxm.parse (abc_string, rOpt, bOpt, fOpt)
    writefile (pad, fnm, '', score, mxl, tOpt)

def writefile (pad, fnm, fnmNum, xmldoc, mxlOpt, tOpt=False):
    ipad, ifnm = os.path.split (fnm)                    # base name of input path is
    if tOpt:
        x = xmldoc.findtext ('work/work-title', 'no_title')
        ifnm = x.replace (',','_').replace ("'",'_').replace ('?','_')
    else:
        ifnm += fnmNum
    xmlstr = fixDoctype (xmldoc)
    if pad:
        if not mxlOpt or mxlOpt in ['a', 'add']:
            outfnm = os.path.join (pad, ifnm + '.xml')  # joined with path from -o option
            outfile = open (outfnm, 'w')
            outfile.write (xmlstr)
            outfile.close ()
            info ('%s written' % outfnm, warn=0)
        if mxlOpt: xml2mxl (pad, ifnm, xmlstr)          # also write a compressed version
    else:
        outfile = sys.stdout
        outfile.write (xmlstr)
        outfile.write ('\n')

def readfile (fnmext, errmsg='read error: '):
    try:
        if fnmext == '-.abc': fobj = stdin  # see python2/3 differences
        else: fobj = open (fnmext, 'rb')
        encoded_data = fobj.read ()
        fobj.close ()
        return encoded_data if type (encoded_data) == uni_type else decodeInput (encoded_data)
    except Exception as e:
        info (errmsg + repr (e) + ' ' + fnmext)
        return None

def expand_abc_include (abctxt):
    ys = []
    for x in abctxt.splitlines ():
        if x.startswith ('%%abc-include') or x.startswith ('I:abc-include'):
            x = readfile (x[13:].strip (), 'include error: ')
        if x != None: ys.append (x)
    return '\n'.join (ys)

abc_header, abc_voice, abc_scoredef, abc_percmap = abc_grammar () # compute grammars only once
mxm = MusicXml ()               # same for instance of MusicXml

def getXmlScores (abc_string, skip=0, num=1, rOpt=False, bOpt=False, fOpt=False): # not used, backwards compatibility
    return [fixDoctype (xml_doc) for xml_doc in
        getXmlDocs (abc_string, skip=0, num=1, rOpt=False, bOpt=False, fOpt=False)]

def getXmlDocs (abc_string, skip=0, num=1, rOpt=False, bOpt=False, fOpt=False): # added by David Randolph
    xml_docs = []
    abctext = expand_abc_include (abc_string)
    fragments = re.split (r'^\s*X:', abctext, flags=re.M)
    preamble = fragments [0]    # tunes can be preceeded by formatting instructions
    tunes = fragments[1:]
    if not tunes and preamble: tunes, preamble = ['1\n' + preamble], ''  # tune without X:
    for itune, tune in enumerate (tunes):
        if itune < skip: continue           # skip tunes, then read at most num tunes
        if itune >= skip + num: break
        tune = preamble + 'X:' + tune       # restore preamble before each tune
        try:                                # convert string abctext -> file pad/fnmNum.xml
            score = mxm.parse (tune, rOpt, bOpt, fOpt)
            ds = list (score.iter ('duration')) # need to iterate twice
            ss = [int (d.text) for d in ds]
            deler = reduce (ggd, ss + [21]) # greatest common divisor of all durations
            for i, d in enumerate (ds): d.text = str (ss [i] // deler)
            for d in score.iter ('divisions'): d.text = str (int (d.text) // deler)
            xml_docs.append (score)
        except ParseException:
            pass         # output already printed
        except Exception as err:
            info ('an exception occurred.\n%s' % err)
    return xml_docs

#----------------
# Main Program
#----------------
if __name__ == '__main__':
    from optparse import OptionParser
    from glob import glob
    import time

    parser = OptionParser (usage='%prog [-h] [-r] [-t] [-b] [-m SKIP NUM] [-o DIR] [-p PFMT] [-z MODE] [--meta MAP] <file1> [<file2> ...]', version='version %d' % VERSION)
    parser.add_option ("-o", action="store", help="store xml files in DIR", default='', metavar='DIR')
    parser.add_option ("-m", action="store", help="skip SKIP (0) tunes, then read at most NUM (1) tunes", nargs=2, type='int', default=(0,1), metavar='SKIP NUM')
    parser.add_option ("-p", action="store", help="pageformat PFMT (mm) = scale (0.75), pageheight (297), pagewidth (210), leftmargin (18), rightmargin (18), topmargin (10), botmargin (10)", default='', metavar='PFMT')
    parser.add_option ("-z", "--mxl", dest="mxl", help="store as compressed mxl, MODE = a(dd) or r(eplace)", default='', metavar='MODE')
    parser.add_option ("-r", action="store_true", help="show whole measure rests in merged staffs", default=False)
    parser.add_option ("-t", action="store_true", help="use tune title as file name", default=False)
    parser.add_option ("-b", action="store_true", help="line break at EOL", default=False)
    parser.add_option ("--meta", action="store", help="map infofields to XML metadata, MAP = R:poet,Z:lyricist,N:...", default='', metavar='MAP')
    parser.add_option ("-f", action="store_true", help="force string/fret allocations for tab staves", default=False)
    options, args = parser.parse_args ()
    if len (args) == 0: parser.error ('no input file given')
    pad = options.o
    if options.mxl and options.mxl not in ['a','add', 'r', 'replace']:
        parser.error ('MODE should be a(dd) or r(eplace), not: %s' % options.mxl)
    if pad:
        if not os.path.exists (pad): os.mkdir (pad)
        if not os.path.isdir (pad): parser.error ('%s is not a directory' % pad)
    if options.p:   # set page formatting values
        try:        # space, page-height, -width, margin-left, -right, -top, -bottom
            mxm.pageFmtCmd = lmap (float, options.p.split (','))
            if len (mxm.pageFmtCmd) != 7: raise ValueError ('-p needs 7 values')
        except Exception as err: parser.error (err)
    for x in options.meta.split (','):
        if not x: continue
        try: field, tag = x.split (':')
        except: parser.error ('--meta: %s cannot be split on colon' % x)
        if field not in 'OAZNGHRBDFSPW': parser.error ('--meta: field %s is no valid ABC field' % field)
        if tag not in mxm.metaTypes: parser.error ('--meta: tag %s is no valid XML creator type' % tag)
        mxm.metaMap [field] = tag
    fnmext_list = []
    for i in args:
        if i == '-': fnmext_list.append ('-.abc')   # represents standard input
        else:        fnmext_list += glob (i)
    if not fnmext_list: parser.error ('none of the input files exist')
    t_start = time.time ()
    for fnmext in fnmext_list:
        fnm, ext = os.path.splitext (fnmext)
        if ext.lower () not in ('.abc'):
            info ('skipped input file %s, it should have extension .abc' % fnmext)
            continue
        if os.path.isdir (fnmext):
            info ('skipped directory %s. Only files are accepted' % fnmext)
            continue
        abctext = readfile (fnmext)
        skip, num = options.m
        xml_docs = getXmlDocs (abctext, skip, num, options.r, options.b, options.f)
        for itune, xmldoc in enumerate (xml_docs):
            fnmNum = '%02d' % (itune + 1) if len (xml_docs) > 1 else ''
            writefile (pad, fnm, fnmNum, xmldoc, options.mxl, options.t)
    info ('done in %.2f secs' % (time.time () - t_start))
