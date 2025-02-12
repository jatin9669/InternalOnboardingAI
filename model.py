from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="TheBloke/Mistral-7B-v0.1-GGUF",
	filename="mistral-7b-v0.1.Q2_K.gguf",
)

output = llm(
	"Once upon a time,",
	max_tokens=512,
	echo=True
)
print(output)