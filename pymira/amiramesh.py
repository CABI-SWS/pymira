# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:40:52 2016

@author: simon

Class for reading and writing Amira mesh files

"""

import numpy as np
import re
from tqdm import tqdm # Progress bar
import sys
import os

class AmiraMesh(object):
    
    def __init__(self):
        self.parameters = None
        self.paramText = None
        self.fileType = None
        self.definitions = None
        self.fields = None
        self.fieldNames = None
        self.fieldOffset = None
        self.fileRead = False
        self.dataFieldCount = 0
        self.header = None
        self.dataTypes = ['float','double','byte','int','long','binary','bool']
        self.filename = None
        self.dir = None
        self.diagnostic = None
        
    def _populate_next_data_field(self,i,curDataMarker,eof=False):
        self.dataFieldCount += 1
        self.fieldOffset.append(i)
        if self.dataFieldCount>1:
            strt = self.fieldOffset[self.dataFieldCount-1]+1
            fin = self.fieldOffset[self.dataFieldCount]-1
            if eof:
                fin += 1
            print(curDataMarker)
            if not self._read_file_data(content,strt,fin,curDataMarker):
                return
        
    def _dataline(self,curLine):
        endPat = '1234567890@ \n\r'
        chk = len([x for x in curLine if x in endPat])==len(curLine) and '@' in curLine
        marker = None
        markerIndex = 0
        if chk:
            mtch = re.search(r'@\w',curLine)
            if mtch is not None:
                try:
                    marker = mtch.group(0)
                    markerIndex = int(marker.replace('@',''))
                except Exception as e:
                    print((repr(e)))
                    return chk,marker,markerIndex
        return chk,marker,markerIndex
        
    def _read_file_data(self,content,strt,fin,curDataMarker,quiet=False):
        
        # Get corresponding field and definition entries
        curField = [x for x in self.fields if x['marker']==curDataMarker][0]
        curDef = [x for x in self.definitions if x['name']==curField['definition']][0]
        
        # Update field entry to include data size and shape
        curField['nentries'] = [int(x) for x in curDef['size']]
        curShape = [int(x) for x in curDef['size']]
        if curField['nelements']>1:
            curShape.append(curField['nelements'])
        curField['shape'] = curShape
            
        # Grab data
        fin = content[strt:].find('@')
        if fin<0:
            fin = len(content)-strt
        curData = content[strt:strt+fin] # -1
        
        # Data type
        if 'ascii' in self.fileType.lower():
            #curData = curData.splitlines()
            curData = [s.strip().split(' ') for s in curData.splitlines() if s]
            comment_line = []
            for k,cnt in enumerate(curData):
                if '#' in cnt:
                    comment_line.append(k)
                else:
                    cnt = [x for x in cnt if x!='']
                    curData[k] = cnt
            if len(comment_line)>0:
                curData = [x for k,x in enumerate(curData) if k not in comment_line]
                
            if curField['type']=='float':
                dtype = np.dtype('f')
            elif curField['type']=='double':
                dtype = np.dtype('d')                     
            elif curField['type']=='int':
                dtype = np.dtype('i')
            elif curField['type']=='long':
                dtype = np.dtype('i')
            elif curField['type']=='byte':
                dtype = np.dtype('byte')
            elif curField['type']=='bool':
                dtype = np.dtype('bool')
            else: # Default to float
                dtype = np.dtype('f')
            try:
                curData = np.asarray(curData) #,dtype=dtype)
                curData = np.reshape(curData,curField['shape'])
                if str(dtype).lower() not in ['byte','bool']:
                    curData = curData.astype('float')
                curField['data'] = curData.astype(dtype)
            except Exception as e:
                if not quiet:
                    fname = curField['name']
                    print(f'Error reading {fname} data: {e}')

        elif 'binary' in self.fileType.lower():
            #curData = self.decode_rle(self,curData,uncompressed_size)

            if curField['encoding']=='raw':
                if curField['type'] != 'byte':
                    raise Exception('Unsupported data type')
                curData = curData.strip('\n\r')
            else:
                enc = curField['encoding']
                num_bytes = int(curField['encoding_length'])
                curData = curData.strip('\n\r')
                assert len(curData)==num_bytes
    
                assert curField['type'] == 'byte'
                uncompressed_size = np.prod(curField['shape'])
                curData = self.decode_rle(curData, uncompressed_size)
            flt_values = np.fromstring(data_strip, dtype=np.uint8) # only tested with bytes

        else:
            raise Exception('File type not currently supported!')
        
        return True
        
    def _check_type(self,str_,type='float'):
        
        # See if a string can be converted into a number
        try:
            if type=='float':
                res = float(str_)
            elif type=='double':
                res = float(str_)
            elif type=='byte':
                res = byte(str_)                
            elif type=='int':
                res = float(str_)
            elif type=='long':
                res = float(str_)
            else:
                return None
            return True
        except:
            return False

    def _field_generator(self,name='',marker='', 
                         definition='',type='',
                         encoding='',encoding_length=0,
                         nelements=0,nentries=0,data=None,shape=None):
        if marker=='':                    
            marker = self.generate_next_marker()
        return {'name':name,'marker':marker,
                'definition':definition,'type':type,
                'encoding':encoding,'encoding_length':encoding_length,
                'nelements':nelements,'nentries':nentries,'data':data,'shape':shape}
                
#    def add_field(self,**kwargs):
#        self.fields.append(self._field_generator(**kwargs))
        
    def read(self,filename,quiet=False):
        
        self.fileRead = False
        self.filename = filename
        import os
        self.dir = os.path.dirname(filename)
        
        self.parameters = []
        self.definitions = []
        self.fileType = None
        self.paramText = []
        self.fields = []
        self.fieldOffset = []
        self.data = []
        self.header = []
        
        inHeader = True
        inParam = False
        curLine = ''
        firstLineFound = False
        nonEmptyLineCount = 0
        self.dataFieldCount = 0
        
        # Remove line returns
        #content = [x.strip('\n') for x in content]
        
        #pbar = tqdm(total=100)
        #nline = len(content)
        
        content = []
        bytesRead = 0
        fileSize = os.path.getsize(filename)
        
        with open(filename,'r') as f:
            i = -1
            while True:
                if inHeader:
                    curLine = f.readline()
                    #print curLine
                    i += 1
                    bytesRead += sys.getsizeof(curLine)
                    curLine = curLine.strip('\n\r')
                
                    if curLine=='' or curLine==' ':
                        pass
                    else:
                        nonEmptyLineCount += 1
                        content.append(curLine)
                    
                    # Check for line definition symbols (if line is short)
                    if len(curLine)<200:
                        defChk = 'define' in curLine.lower()
                        hashChk = '#' in curLine
                        atChk = '@' in curLine
                        paramChk = 'parameters' in curLine.lower()
                        dataSectionChk,curDataMarker,curMarkerIndex = self._dataline(curLine)
                    else:
                        defChk = False
                        hashChk = False
                        atChk = False
                        paramChk = False
                        dataSectionChk = False
                        
                    if nonEmptyLineCount==1 and firstLineFound is False:
                        # Look for magic
                        firstLineFound = True
                        ptn = re.compile("\s*#\s*amiramesh")
                        magicMtch = ptn.match(curLine.lower()) is not None
                        if not magicMtch:
                            raise Exception('Error: Not a supported Amira file!')
                        spl = curLine.split('AmiraMesh')
                        self.fileType = ''.join(spl[1:]).strip()
                        
                    # Parameter line
                    if paramChk:
                        self.parameters = []
                        inParam = True
                        bracketCount = 0
                        
                    if inParam:
                        self.paramText.append(curLine)
                        spl = str.split(curLine,' ')
                        spl = [x for x in spl if x not in ['','\t']]
                        nspl = len(spl)
                        if '{' in curLine:
                            bracketCount += 1
                        if '}' in curLine:
                            bracketCount -= 1
                        #print 'Bracket: ',bracketCount
                        if nspl>1 and '{' not in curLine and '}' not in curLine: # Nested parameter structure not yet supported
                        
                            curParam = spl[0]
                            if '"' in curLine:
                                val = ' '.join(spl[1:])
                                val = val.replace('"','').rstrip()
                                self.parameters.append({'parameter':curParam,'value':val})
                            elif curParam=='ContentType':
                                spl2 = str.split(spl[1],'\"')
                                curParamVal = spl2[0]
                                self.parameters.append({'parameter':curParam,'value':curParamVal})                        
                            elif curParam=='TransformationMatrix':
                                matr = [float(re.sub(r'\W+','',x)) for x in spl[1:]]
                                self.parameters.append({'parameter':curParam,'value':matr})                        
                            elif curParam=='BoundingBox':
                                bbox = [float(re.sub(r'\W+','',x)) for x in spl[1:]]
                                self.parameters.append({'parameter':curParam,'value':bbox})                        
                            else:
                                val = spl[1:]
                                self.parameters.append({'parameter':curParam,'value':val})
        
                        if bracketCount==0:
                            inParam = False
                                
                    if defChk and not inParam: # Definition line
                        spl = str.split(curLine, ' ')
                        spl = [x for x in spl if x]
                        nspl = len(spl)
                        if nspl>2:
                            self.definitions.append({'name':spl[1],'size':[int(x) for x in spl[2:]]})
                        else:
                            raise Exception('Error: Unexpected formatting on definition line!')
                            #return False
                            
                    if atChk and not inParam and not dataSectionChk: # Field line
                        spl = curLine.split(' ')
                        spl = [x for x in spl if x!='']
                        
                        defMtch = [x for x in self.definitions if x['name']==spl[0]]
                        
                        if len(defMtch)>0:
                            curLineNoWS = curLine.replace(' ','')
                            spl = re.split(r"[{}]+", curLineNoWS)
                            spl = [x for x in spl if x]
                            
                            fieldDef = spl[0].strip()
                            fieldType = spl[1].strip()

                            m = re.split(r"\[(\w+)\]", fieldType)
                            try:
                                if len(m)==3:
                                    fieldDataType = m[0].strip()
                                    if fieldDataType not in self.dataTypes:
                                         raise Exception('Data type not recognised in field line: ',curLine)
                                         #return False
                                    nel = int(m[1])
                                    fieldName = m[2].strip()
                                else:
                                    fieldDataType = [x for x in self.dataTypes if x in fieldType]
                                    if len(fieldDataType)==0:
                                        raise Exception('Data type not recognised in field line: ',curLine)
                                    fieldDataType = fieldDataType[0]
                                    fieldName = fieldType.replace(fieldDataType,'',1).strip()
                                    nel = 1
                            except Exception as e:
                                breakpoint()
                                raise Exception(e)
                                #breakpoint()
                                #return None

                            if len(spl)==3:
                                marker = spl[2]
                                spl2 = re.split(r"[()]+", spl[2])
                                spl2 = [x for x in spl2 if x]
                                if len(spl2)>1:
                                    marker = spl2[0].strip()
                                    fieldTypeInfo = spl2[1].split(',')
                                    fieldTypeInfo = [x for x in fieldTypeInfo if x]
                                    assert len(fieldTypeInfo)==2
                                else:
                                    fieldTypeInfo = [None,None]
                            # RLE (and other?) files have an extra section in parentheses here
                            elif len(spl)==4:
                                spl2 = re.split(r"[()]+", spl[3])
                                spl2 = [x for x in spl2 if x]
                                marker = spl2[0].strip()
                                if len(spl2)>1:
                                    fieldTypeInfo = spl2[1].split(',')
                                    fieldTypeInfo = [x for x in fieldTypeInfo if x]
                                    assert len(fieldTypeInfo)==2
                            else:
                                raise Exception('Data type not recognised in field line: ',curLine)
                                #return None
                            
#                            def_ = [x for x in self.definitions if x['name']==fieldDef]
#                            assert len(def_)==1
#                            def_ = def_[0]
#                            if getattr(def_,'type',None) is None:
#                                def_['type'] = fieldDataType
#                                def_['nelements'] = nel
#                            else:
#                                assert def_['type']==fieldDataType
#                                assert def_['nelements']==nel
                            self.fields.append(self._field_generator(name=fieldName,marker=marker,
                                                 definition=fieldDef,type=fieldDataType,
                                                 encoding=fieldTypeInfo[0],encoding_length=fieldTypeInfo[1],
                                                 nelements=nel,nentries=None))
                        else:
                            pass
                    
                    if dataSectionChk: # Come out of header and move on to data
                        #self.fieldOffset.append(i)         
                        inHeader = False
                        self.header = content[0:-2]
                        self.data = content[-1]
                        content = []
                    
                else:
#                    if 'binary' in self.fileType.lower():
#                        import pdb
#                        pdb.set_trace()
#                        f.close()
#                        hdrEnd = sys.getsizeof(self.header)
#                        with open(filename,'rb') as f:
#                            f.seek(hdrEnd+1)
#                            self.data = f.read() 
#                    else:
                    if True:
                        self.data += f.read()
                    bytesRead += sys.getsizeof(self.data)
                    if bytesRead<fileSize:
                        pass
                        #import pdb
                        #pdb.set_trace()
                        #raise Exception('Not all of file was read! Only {} of {}'.format(bytesRead,fileSize))
                    break

        self.fieldRange = []
        for curField in self.fields:
            mrk = curField['marker']
            mInd = self.data.find(mrk+'\n')
            if mInd==-1:
                mInd = self.data.find(mrk)
            mInd += len(mrk)
            self.fieldRange.append(mInd)
        self.fieldRange.append(len(self.data))

        for i,curField in enumerate(self.fields):
            self._read_file_data(self.data,self.fieldRange[i],self.fieldRange[i+1],curField['marker'],quiet=quiet)
            
        # Remove any null data fields
        self.fields = [x for x in self.fields if x['data'] is not None]

        # Populate field names            
        self.fieldNames = [x['name'] for x in self.fields]

        # Populate the last field
        #self._populate_next_data_field(i,curDataMarker,eof=True)

        self.fileRead = True
        
        return True
        
    def add_definition(self,name,size):
        if type(size) is not list:
            size = [size]
        if self.definitions is None:
            self.definitions = []
        self.definitions.append({'name':name,'size':size})
        
    def add_parameter(self,name,value):
        if self.parameters is None:
            self.parameters = []
        self.parameters.append({'parameter':name,'value':value})
        
    def add_field(self,**kwargs):
        if self.fields is None:
            self.fields = []
        if self.fieldNames is None:
            self.fieldNames = []
        field = self._field_generator(**kwargs)
        self.fields.append(field)
        if self.fieldNames is not None:
            self.fieldNames.append(field['name'])
        
        if 'data' in kwargs.keys() and kwargs['data'] is not None:
            self.set_data(kwargs['data'],name=field['name'])
        
        return field
        
    def write_json(self,filename):
    
        import json

        o = dict()
        # AmiraMesh object data held in fields by name:
        # 'VertexCoordinates', 'EdgeConnectivity', 'NumEdgePoints', 'EdgePointCoordinates', 'thickness'
        # Dislike the naming of thickness, so capitalize.
        # Data stored as numpy array, so call tolist()
        for field in self.fields:
            name = field['name']
            if name.lower()=='radii':
                name = 'radius'
            name = name[0].upper() + name[1:]
            if field['data'] is not None:
                o[name] = field['data'].tolist()

        with open(filename, 'w') as handle:
            json.dump(o, handle, indent=4)
        
    def write(self,filename):
        with open(filename, 'w') as f:
            f.write('# AmiraMesh {}\n'.format(self.fileType))
            f.write('\n')
            
            # Definition section
            for d in self.definitions:
                dSize = d['size'][0]
                if type(dSize) is list:
                    dSize = ' '.join([str(ds) for ds in dSize])
                f.write('define {} {}\n'.format(d['name'],dSize))
            f.write('\n')
            
            # Parameter section
            f.write('Parameters {\n')
            nparam = len(self.parameters)
            for pi,p in enumerate(self.parameters):
                if type(p['value']) not in [list,np.ndarray]:# or len(p['value'])==1:
                    curLine = '   {} \"{}\"'.format(p['parameter'],p['value'])
                else:
                    pStr = ' '.join([str(b) for b in p['value']])
                    curLine = '   {} {}'.format(p['parameter'],pStr)
                if pi<nparam-1:
                    curLine += ',\n'
                else:
                    curLine += '\n'
                f.write(curLine)
            f.write('}\n')
            f.write('\n')
            
            # Data definition section
            for d in self.fields:
                if d['nelements']>1:
                    f.write('{0} {{ {1}[{2}] {3} }} {4} \n'.format(d['definition'],d['type'],d['nelements'],d['name'],d['marker']))
                else:
                    f.write('{0} {{ {1} {2} }} {3} \n'.format(d['definition'],d['type'],d['name'],d['marker']))
            f.write('\n')
            
            # Data section
            f.write('# Data section\n')
            for d in self.fields:
                f.write('{}\n'.format(d['marker']))
                data = d['data']
                name = d['name']
                if data.ndim==1:
                    for j in range(data.shape[0]):
                        if data[j] is None:
                            breakpoint()
                        data[j].tofile(f, sep=" ", format="%s")
                        f.write('\n')
                elif data.ndim==2:
                    for j in range(data.shape[0]):
                        data[j,:].tofile(f, sep=" ", format="%s")
                        f.write('\n')
                else:
                    for j in range(data.shape[2]):
                        for k in range(data.shape[1]):
                            for l in range(data.shape[0]):
                                data[l,k,j].tofile(f, sep=" ", format="%s")
                                f.write('\n')
                f.write('\n')
        
#    def add_field(self,fieldDict):
#        markers = [int(x['marker'].replace('@','')) for x in self.fields]
#        fieldDict['marker'] = '@{}'.format(max(markers)+1)
#        defNames = [x['name'] for x in self.definitions]
#        assert fieldDict['definition'] in defNames
#        # TO DO: add check for consistency with definition...
#        
#        self.fields.append(fieldDict)
#        self.fieldNames.append(fieldDict['name'])
#        self.fieldRange.append(-1)
        
    def decode_rle(self,d,uncompressed_size):
        # based on decode.rle from nat's amiramesh-io.R
        d = np.fromstring(d, dtype=np.uint8 )
        rval = np.empty( (uncompressed_size,), dtype=np.uint8 )
        bytes_read = 0
        filepos = 0
        while bytes_read < uncompressed_size:
            x=d[filepos]
            if x==0:
                raise ValueError('x is 0 at %d'%filepos)
            filepos += 1
    
            if x > 0x7f:
                x = (x & 0x7f)
                mybytes=d[filepos:filepos+x]
                filepos += x
            else:
                mybytes=np.repeat(d[filepos],x)
                filepos += 1
            rval[bytes_read:bytes_read+len(mybytes)] = mybytes
            bytes_read += len(mybytes)
        return rval

    def get_parameter_value(self,paramName):
        #if not self.fileRead:
        #    return None
        
        val = [x['value'] for x in self.parameters if x['parameter']==paramName]
        if len(val)==0:
            return None
        return val[0]
        
    def get_field(self,name=None,marker=None):
        #if not self.fileRead:
        #    return None
        
        if name is not None:
            val = [x for x in self.fields if x['name']==name]
            if len(val)==0:
                return None
            return val[0]
        elif marker is not None:
            val = [x for x in self.fields if x['marker']==marker]
            if len(val)==0:
                return None
            return val[0]
            
    def get_field_names(self):
        return [x['name'] for x in self.fields]
            
    def get_data(self,*args,**kwargs):
        f = self.get_field(*args,**kwargs)
        if f is None:
            return None
        else:
            return f['data']
            
    def set_data(self,data,**kwargs):
        f = self.get_field(**kwargs)
        if f is not None:
            f['data'] = data
            f['shape'] = data.shape
            try:
                f['nentries'] = [len(data)]
            except:
                f['nentries'] = [1]
            if self.diagnostic is not None:
                self.diagnostic(func='set_data',a={'data':data,'field':f})
            
    def get_definition(self,name):
        defin = [x for x in self.definitions if x['name'].lower()==name.lower()]
        if len(defin)==0:
            return None
        else:
            return defin[0]
            
    def get_definition_names(self):
        return [x['name'] for x in self.definitions]
            
    def set_definition_size(self,name,size):
        defin = self.get_definition(name)
        if defin is None:
            return
        if type(size) is not list:
            size = [size]
        defin['size'] = size
        
    def generate_next_marker(self):
        marker_indices = [int(f['marker'].replace('@','')) for f in self.fields]
        mx = max(marker_indices)
        return '@{}'.format(mx+1)
            
    def close_error(self,filename):
        
        fileSize = os.path.getsize(filename) # file size in bytes
        bytesRead = 0
        content = []
        count = 0
        try:
            while True:
                count += 1
                with open(filename,'r') as f:
                    f.seek(bytesRead+1)
                    newContent = f.read()
                    content.append(newContent)
                    bytesRead += sys.getsizeof(newContent)
                    print((count,' Total read:',bytesRead))
        except Exception as e:
            print(('Exception: ',e))
            
        import pdb
        pdb.set_trace()
          
        print(('File size:',fileSize)) 
        print(('% read = ',bytesRead*100./float(fileSize)))
        print(('count: ',count)) 
