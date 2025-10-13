from transformers import LlamaTokenizerFast, PreTrainedTokenizer
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import argparse
from typing import List

def mask_multichar_chinese_tokens(tokenizer: PreTrainedTokenizer):
    """Create a tokenizer wrapper that converts multi-character Chinese tokens to single characters.
    
    This function creates a wrapper around the provided tokenizer that automatically
    splits multi-character Chinese tokens into individual characters. This is useful
    for ensuring consistent tokenization of Chinese text.
    
    Args:
        tokenizer: The base tokenizer to wrap
        
    Returns:
        A CharTokenizerWrapper instance that handles multi-character Chinese tokens
        
    Example:
        >>> from transformers import LlamaTokenizerFast
        >>> tokenizer = LlamaTokenizerFast.from_pretrained("path/to/tokenizer")
        >>> wrapped_tokenizer = mask_multichar_chinese_tokens(tokenizer)
        >>> tokens = wrapped_tokenizer("你好世界")
    """
    # Pre-compute multi-character tokens (length >= 2, pure Chinese characters)
    multichar_tokens = {
        token for token in tokenizer.vocab.keys() 
        if len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
    }

    class CharTokenizerWrapper:
        """Wrapper class for tokenizers that handles multi-character Chinese tokens.
        
        This wrapper automatically splits multi-character Chinese tokens into
        individual characters while preserving the original tokenizer's interface.
        """
        
        def __init__(self, base_tokenizer: PreTrainedTokenizer) -> None:
            """Initialize the wrapper with a base tokenizer.
            
            Args:
                base_tokenizer: The tokenizer to wrap
            """
            self.tokenizer = base_tokenizer
            self.multichar_tokens = multichar_tokens

        def tokenize(self, text: str, **kwargs) -> List[str]:
            """Tokenize text and split multi-character Chinese tokens into single characters.
            
            Args:
                text: Input text to tokenize
                **kwargs: Additional arguments passed to the base tokenizer
                
            Returns:
                List of processed tokens with multi-character Chinese tokens split
                
            Example:
                >>> wrapper = CharTokenizerWrapper(tokenizer)
                >>> tokens = wrapper.tokenize("你好世界")
                >>> # Returns ["你", "好", "世", "界"] instead of ["你好", "世界"]
            """
            if not isinstance(text, str):
                raise TypeError(f"Expected string input, got {type(text)}")
                
            tokens = self.tokenizer.tokenize(text, **kwargs)
            processed = []
            
            for token in tokens:
                # Remove possible subword prefix
                clean_token = token.replace("▁", "")

                if clean_token in self.multichar_tokens:
                    # Split multi-character token into single characters
                    chars = list(clean_token)
                    processed.extend(chars)
                else:
                    processed.append(token)
                    
            return processed

        def __call__(self, text: str, **kwargs) -> List[int]:
            """Call the tokenizer and return token IDs.
            
            This method provides the same interface as the original tokenizer
            but with multi-character Chinese token handling.
            
            Args:
                text: Input text to tokenize
                **kwargs: Additional arguments passed to the base tokenizer
                
            Returns:
                List of token IDs
                
            Raises:
                TypeError: If input is not a string
                ValueError: If tokenization fails
            """
            try:
                tokens = self.tokenize(text, **kwargs)
                result = self.tokenizer.convert_tokens_to_ids(tokens)
                return result
            except Exception as e:
                raise ValueError(f"Tokenization failed: {str(e)}") from e

    return CharTokenizerWrapper(tokenizer)

class Tokenizer_Http():

    def __init__(self):
        tokenizer = LlamaTokenizerFast.from_pretrained("./VoxCPM-0.5B")
        self.tokenizer = mask_multichar_chinese_tokens(tokenizer)

    def encode(self, prompt):
    
        token_ids = self.tokenizer(prompt)
        return token_ids

    # @property
    # def bos_id(self):
    #     return self.tokenizer.bos_token_id

    @property
    def eos_id(self):
        return 1773
    
    # @property
    # def bos_token(self):
    #     return self.tokenizer.bos_token

    @property
    def eos_token(self):
        return "<|eot_id|>"


tokenizer = Tokenizer_Http()

# print(tokenizer.bos_id, tokenizer.bos_token, tokenizer.eos_id, tokenizer.eos_token)
print(tokenizer.encode("hello world"))


class Request(BaseHTTPRequestHandler):
    #通过类继承，新定义类
    timeout = 5
    server_version = 'Apache'

    def do_GET(self):
        print(self.path)
        #在新类中定义get的内容（当客户端向该服务端使用get请求时，本服务端将如下运行）
        self.send_response(200)
        self.send_header("type", "get")  #设置响应头，可省略或设置多个
        self.end_headers()

        if self.path == '/bos_id':
            bos_id = tokenizer.bos_id
            # print(bos_id)
            # to json
            if bos_id is None:
                msg = json.dumps({'bos_id': -1})
            else:
                msg = json.dumps({'bos_id': bos_id})
        elif self.path == '/eos_id':
            eos_id = tokenizer.eos_id
            if eos_id is None:
                msg = json.dumps({'eos_id': -1})
            else:
                msg = json.dumps({'eos_id': eos_id})
        else:
            msg = 'error'

        print(msg)
        msg = str(msg).encode()  #转为str再转为byte格式

        self.wfile.write(msg)  #将byte格式的信息返回给客户端

    def do_POST(self):
        #在新类中定义post的内容（当客户端向该服务端使用post请求时，本服务端将如下运行）
        data = self.rfile.read(int(
            self.headers['content-length']))  #获取从客户端传入的参数（byte格式）
        data = data.decode()  #将byte格式转为str格式

        self.send_response(200)
        self.send_header("type", "post")  #设置响应头，可省略或设置多个
        self.end_headers()

        if self.path == '/encode':
            req = json.loads(data)
            prompt = req['text']

            token_ids = tokenizer.encode(prompt)
            if token_ids is None:
                msg = json.dumps({'token_ids': -1})
            else:
                msg = json.dumps({'token_ids': token_ids})

        # elif self.path == '/decode':
        #     req = json.loads(data)
        #     token_ids = req['token_ids']
        #     text = tokenizer.decode(token_ids)
        #     if text is None:
        #         msg = json.dumps({'text': ""})
        #     else:
        #         msg = json.dumps({'text': text})
        else:
            msg = 'error'
        print(msg)
        msg = str(msg).encode()  #转为str再转为byte格式

        self.wfile.write(msg)  #将byte格式的信息返回给客户端


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument('--host', type=str, default='localhost')
    args.add_argument('--port', type=int, default=12345)
    args = args.parse_args()

    host = (args.host, args.port)  #设定地址与端口号，'localhost'等价于'127.0.0.1'
    print('http://%s:%s' % host)
    server = HTTPServer(host, Request)  #根据地址端口号和新定义的类，创建服务器实例
    server.serve_forever()  #开启服务
