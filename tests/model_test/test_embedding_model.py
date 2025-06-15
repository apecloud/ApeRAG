# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Embedding Model Test Script

This script tests all available embedding models in the deployed ApeRAG system.
It's designed to be run after system deployment to verify which models are actually
functional, considering factors like API key configuration, provider availability, etc.

Usage:
    python tests/model_test/test_embedding_model.py

The script will:
1. Fetch all available embedding models from /api/v1/available_models
2. Test each model by calling /api/v1/embeddings with a test sentence
3. Generate a JSON report with test results

Environment Variables:
    APERAG_API_URL: Base URL for the ApeRAG API (default: http://localhost:8000)
    APERAG_USERNAME: Username for authentication
    APERAG_PASSWORD: Password for authentication
    EMBEDDING_TEST_TEXT: Custom text for embedding test (optional)
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx

# --- Configuration ---
API_BASE_URL = os.getenv("APERAG_API_URL", "http://localhost:8000")
USERNAME = os.getenv("APERAG_USERNAME", "user@nextmail.com")
PASSWORD = os.getenv("APERAG_PASSWORD", "123456")
DEFAULT_EMBEDDING_TEST_TEXT_SHORT = """The rapid advancements in artificial intelligence are reshaping various aspects of our lives. 人工智能的迅猛发展正在重塑我们生活的方方面面"""
DEFAULT_EMBEDDING_TEST_TEXT = """The rapid advancements in artificial intelligence are reshaping various aspects of our lives, from how we work and communicate to how we learn and create. Machine learning, a subset of AI, has enabled the development of sophisticated algorithms capable of processing vast amounts of data, identifying patterns, and making predictions with remarkable accuracy. This has led to breakthroughs in fields such as natural language processing, computer vision, and robotics. For instance, large language models (LLMs) are now able to generate coherent and contextually relevant text, translate languages with impressive fluidity, and even assist in complex problem-solving. The ethical implications of AI are also a significant area of discussion. Questions around data privacy, algorithmic bias, and the potential impact on employment are being actively debated by researchers, policymakers, and the general public. Ensuring that AI development is guided by principles of fairness, transparency, and accountability is crucial for harnessing its potential for societal good. The integration of AI into industries like healthcare, finance, and manufacturing is already yielding significant efficiencies and innovations. In healthcare, AI assists in diagnosing diseases, personalizing treatment plans, and accelerating drug discovery. In finance, it helps detect fraud, manage risks, and optimize investment strategies. The future of AI promises even more transformative changes, with ongoing research focusing on areas like explainable AI, artificial general intelligence, and human-AI collaboration. Understanding and adapting to these changes will be key for individuals and organizations alike to thrive in an increasingly AI-driven world. The journey of AI development is still in its early stages, yet its profound impact is already undeniable, prompting us to consider both its immense opportunities and its inherent challenges.
人工智能的迅猛发展正在重塑我们生活的方方面面，从工作和沟通方式到学习和创造过程。机器学习作为人工智能的一个分支，已经促使了复杂算法的开发，这些算法能够处理海量数据，识别模式，并以惊人的准确性进行预测。这在自然语言处理、计算机视觉和机器人技术等领域取得了突破性进展。例如，大型语言模型（LLMs）现在能够生成连贯且符合上下文的文本，以令人印象深刻的流畅性翻译语言，甚至协助解决复杂问题。人工智能的伦理影响也是一个重要的讨论领域。关于数据隐私、算法偏见以及对就业的潜在影响等问题，研究人员、政策制定者和公众都在积极辩论。确保人工智能的发展以公平、透明和负责任的原则为指导，对于利用其潜力造福社会至关重要。人工智能与医疗保健、金融和制造业等行业的融合已经带来了显著的效率提升和创新。在医疗保健领域，人工智能协助诊断疾病、个性化治疗方案并加速药物发现。在金融领域，它有助于检测欺诈、管理风险和优化投资策略。人工智能的未来预示着更具变革性的变化，目前的研究重点领域包括可解释人工智能、通用人工智能和人机协作。理解并适应这些变化对于个人和组织在日益由人工智能驱动的世界中蓬勃发展至关重要。人工智能发展之旅仍处于早期阶段，但其深远影响已不容否认，促使我们同时思考其巨大的机遇和固有的挑战。
人工知能の急速な進歩は、私たちの生活の様々な側面を再構築しています。それは仕事やコミュニケーションの方法から、学習や創造のプロセスに至るまで多岐にわたります。AIのサブセットである機械学習は、膨大な量のデータを処理し、パターンを識別し、驚くべき精度で予測を行うことができる洗練されたアルゴリズムの開発を可能にしました。これにより、自然言語処理、コンピュータービジョン、ロボティクスなどの分野で画期的な進歩がもたらされました。例えば、大規模言語モデル（LLMs）は現在、一貫性があり文脈に即したテキストを生成し、驚くほど流暢に言語を翻訳し、さらには複雑な問題解決を支援することも可能です。AIの倫理的影響も重要な議論の領域です。データプライバシー、アルゴリズムバイアス、雇用への潜在的な影響に関する問題は、研究者、政策立案者、一般市民によって活発に議論されています。AI開発が公平性、透明性、説明責任の原則によって導かれることを確実にすることは、その社会的な利益のための可能性を活用するために不可欠です。AIの医療、金融、製造などの産業への統合は、すでに著しい効率と革新を生み出しています。医療分野では、AIは疾病の診断、治療計画の個別化、および新薬開発の加速を支援します。金融分野では、詐欺の検出、リスク管理、投資戦略の最適化に役立ちます。AIの未来は、より変革的な変化を約束しており、現在進行中の研究は、説明可能なAI、汎用人工知能、人間とAIのコラボレーションなどの分野に焦点を当てています。これらの変化を理解し適応することは、個人にとっても組織にとっても、ますますAI駆動型となる世界で成功するために重要です。AI開発の旅はまだ初期段階にありますが、その深遠な影響はすでに否定できません。それは私たちに、その計り知れない機会と固有の課題の両方を考慮するよう促しています。
إن التطورات السريعة في الذكاء الاصطناعي تعيد تشكيل جوانب مختلفة من حياتنا، من كيفية عملنا وتواصلنا إلى كيفية تعلمنا وإبداعنا. لقد أتاح التعلم الآلي، وهو فرع من فروع الذكاء الاصطناعي، تطوير خوارزميات متطورة قادرة على معالجة كميات هائلة من البيانات، وتحديد الأنماط، وتقديم تنبؤات بدقة ملحوظة. وقد أدى ذلك إلى إنجازات في مجالات مثل معالجة اللغة الطبيعية، ورؤية الكمبيوتر، والروبوتات. على سبيل المثال، أصبحت النماذج اللغوية الكبيرة (LLMs) الآن قادرة على توليد نص متماسك وذو صلة بالسياق، وترجمة اللغات بسلاسة مذهلة، وحتى المساعدة في حل المشكلات المعقدة. تعد الآثار الأخلاقية للذكاء الاصطناعي أيضًا مجالًا مهمًا للمناقشة. تتم مناقشة الأسئلة المتعلقة بخصوصية البيانات، والتحيز الخوارزمي، والتأثير المحتمل على التوظيف بنشاط من قبل الباحثين، وصناع السياسات، والجمهور العام. يعد ضمان توجيه تطوير الذكاء الاصطناعي بمبادئ العدالة والشفافية والمساءلة أمرًا بالغ الأهمية لتسخير إمكاناته من أجل الخير الاجتماعي. إن دمج الذكاء الاصطناعي في صناعات مثل الرعاية الصحية، والتمويل، والتصنيع يحقق بالفعل كفاءات وابتكارات كبيرة. في الرعاية الصحية، يساعد الذكاء الاصطناعي في تشخيص الأمراض، وتخصيص خطط العلاج، وتسريع اكتشاف الأدوية. في التمويل، يساعد في اكتشاف الاحتيال، وإدارة المخاطر، وتحسين استراتيجيات الاستثمار. يعد مستقبل الذكاء الاصطناعي بمزيد من التغييرات التحويلية، حيث تركز الأبحاث الجارية على مجالات مثل الذكاء الاصطناعي القابل للتفسير، والذكاء الاصطناعي العام، والتعاون بين البشر والذكاء الاصطناعي. سيكون فهم هذه التغييرات والتكيف معها أمرًا أساسيًا للأفراد والمنظمات على حد سواء للازدهار في عالم يعتمد بشكل متزايد على الذكاء الاصطناعي. لا تزال رحلة تطوير الذكاء الاصطناعي في مراحلها المبكرة، ومع ذلك، فإن تأثيرها العميق لا يمكن إنكاره بالفعل، مما يدفعنا إلى النظر في فرصها الهائلة وتحدياتها المتأصلة.
Les progrès rapides de l'intelligence artificielle transforment divers aspects de nos vies, de notre façon de travailler et de communiquer à notre façon d'apprendre et de créer. L'apprentissage automatique, un sous-ensemble de l'IA, a permis le développement d'algorithmes sophistiqués capables de traiter de vastes quantités de données, d'identifier des schémas et de faire des prédictions avec une précision remarquable. Cela a conduit à des percées dans des domaines tels que le traitement du langage naturel, la vision par ordinateur et la robotique. Par exemple, les grands modèles de langage (LLM) sont désormais capables de générer des textes cohérents et contextuellement pertinents, de traduire des langues avec une fluidité impressionnante et même d'aider à la résolution de problèmes complexes. Les implications éthiques de l'IA sont également un domaine de discussion important. Des questions concernant la confidentialité des données, les biais algorithmiques et l'impact potentiel sur l'emploi sont activement débattues par les chercheurs, les décideurs politiques et le grand public. S'assurer que le développement de l'IA est guidé par des principes d'équité, de transparence et de responsabilité est crucial pour exploiter son potentiel au service du bien social. L'intégration de l'IA dans des industries telles que la santé, la finance et la fabrication génère déjà des gains d'efficacité et des innovations significatifs. Dans le domaine de la santé, l'IA aide à diagnostiquer les maladies, à personnaliser les plans de traitement et à accélérer la découverte de médicaments. Dans la finance, elle aide à détecter la fraude, à gérer les risques et à optimiser les stratégies d'investissement. L'avenir de l'IA promet des changements encore plus transformateurs, les recherches en cours se concentrant sur des domaines tels que l'IA explicable, l'intelligence artificielle générale et la collaboration homme-IA. Comprendre et s'adapter à ces changements sera essentiel pour les individus et les organisations afin de prospérer dans un monde de plus en plus axé sur l'IA. Le parcours du développement de l'IA n'en est qu'à ses débuts, mais son impact profond est déjà indéniable, nous incitant à considérer à la fois ses immenses opportunités et ses défis inhérents.
Los rápidos avances en inteligencia artificial están remodelando varios aspectos de nuestras vidas, desde cómo trabajamos y nos comunicamos hasta cómo aprendemos y creamos. El aprendizaje automático, un subconjunto de la IA, ha permitido el desarrollo de algoritmos sofisticados capaces de procesar grandes cantidades de datos, identificar patrones y hacer predicciones con una precisión notable. Esto ha llevado a avances en campos como el procesamiento del lenguaje natural, la visión por computadora y la robótica. Por ejemplo, los grandes modelos de lenguaje (LLM) ahora son capaces de generar texto coherente y contextualmente relevante, traducir idiomas con una fluidez impresionante e incluso ayudar en la resolución de problemas complejos. Las implicaciones éticas de la IA también son un área importante de discusión. Las preguntas sobre la privacidad de los datos, el sesgo algorítmico y el impacto potencial en el empleo están siendo debatidas activamente por investigadores, formuladores de políticas y el público en general. Asegurar que el desarrollo de la IA se guíe por principios de equidad, transparencia y rendición de cuentas es crucial para aprovechar su potencial para el bien social. La integración de la IA en industrias como la atención médica, las finanzas y la fabricación ya está produciendo eficiencias e innovaciones significativas. En la atención médica, la IA ayuda a diagnosticar enfermedades, personalizar planes de tratamiento y acelerar el descubrimiento de fármacos. En finanzas, ayuda a detectar fraudes, gestionar riesgos y optimizar estrategias de inversión. El futuro de la IA promete cambios aún más transformadores, con investigaciones en curso centradas en áreas como la IA explicable, la inteligencia artificial general y la colaboración entre humanos e IA. Comprender y adaptarse a estos cambios será clave para que tanto individuos como organizaciones prosperen en un mundo cada vez más impulsado por la IA. El camino del desarrollo de la IA aún se encuentra en sus primeras etapas, sin embargo, su profundo impacto ya es innegable, lo que nos impulsa a considerar tanto sus inmensas oportunidades como sus desafíos inherentes.
I rapidi progressi nell'intelligenza artificiale stanno rimodellando vari aspetti delle nostre vite, da come lavoriamo e comunichiamo a come impariamo e creiamo. L'apprendimento automatico, un sottoinsieme dell'IA, ha permesso lo sviluppo di algoritmi sofisticati in grado di elaborare vaste quantità di dati, identificare schemi e fare previsioni con notevole precisione. Ciò ha portato a scoperte in campi come l'elaborazione del linguaggio naturale, la visione artificiale e la robotica. Ad esempio, i grandi modelli linguistici (LLM) sono ora in grado di generare testo coerente e contestualmente pertinente, tradurre lingue con impressionante fluidità e persino assistere nella risoluzione di problemi complessi. Anche le implicazioni etiche dell'IA sono un'area di discussione significativa. Questioni riguardanti la privacy dei dati, i bias algoritmici e il potenziale impatto sull'occupazione sono attivamente dibattute da ricercatori, politici e dal pubblico in generale. Garantire che lo sviluppo dell'IA sia guidato da principi di equità, trasparenza e responsabilità è cruciale per sfruttare il suo potenziale a beneficio sociale. L'integrazione dell'IA in settori come la sanità, la finanza e la produzione sta già producendo significative efficienze e innovazioni. Nell'assistenza sanitaria, l'IA assiste nella diagnosi delle malattie, nella personalizzazione dei piani di trattamento e nell'accelerazione della scoperta di farmaci. Nella finanza, aiuta a rilevare frodi, gestire i rischi e ottimizzare le strategie di investimento. Il futuro dell'IA promette cambiamenti ancora più trasformativi, con la ricerca in corso che si concentra su aree come l'IA spiegabile, l'intelligenza artificiale generale e la collaborazione uomo-IA. Comprendere e adattarsi a questi cambiamenti sarà fondamentale per individui e organizzazioni per prosperare in un mondo sempre più guidato dall'IA. Il percorso di sviluppo dell'IA è ancora nelle sue fasi iniziali, eppure il suo profondo impatto è già innegabile, spingendoci a considerare sia le sue immense opportunità che le sue sfide intrinseche.
Быстрые достижения в области искусственного интеллекта меняют различные аспекты нашей жизни, от того, как мы работаем и общаемся, до того, как мы учимся и творим. Машинное обучение, подмножество ИИ, позволило разработать сложные алгоритмы, способные обрабатывать огромные объемы данных, выявлять закономерности и делать прогнозы с поразительной точностью. Это привело к прорывам в таких областях, как обработка естественного языка, компьютерное зрение и робототехника. Например, большие языковые модели (LLM) теперь способны генерировать связный и контекстно релевантный текст, переводить языки с впечатляющей плавностью и даже помогать в решении сложных задач. Этические последствия ИИ также являются важной областью для обсуждения. Вопросы, касающиеся конфиденциальности данных, алгоритмической предвзятости и потенциального влияния на занятость, активно обсуждаются исследователями, политиками и широкой общественностью. Обеспечение того, чтобы развитие ИИ руководствовалось принципами справедливости, прозрачности и подотчетности, имеет решающее значение для использования его потенциала на благо общества. Интеграция ИИ в такие отрасли, как здравоохранение, финансы и производство, уже приносит значительную эффективность и инновации. В здравоохранении ИИ помогает в диагностике заболеваний, персонализации планов лечения и ускорении разработки лекарств. В финансах он помогает обнаруживать мошенничество, управлять рисками и оптимизировать инвестиционные стратегии. Будущее ИИ обещает еще более преобразующие изменения, при этом текущие исследования сосредоточены на таких областях, как объяснимый ИИ, искусственный общий интеллект и сотрудничество человека с ИИ. Понимание и адаптация к этим изменениям будет ключом к процветанию как для отдельных лиц, так и для организаций в мире, все более управляемом ИИ. Путь развития ИИ все еще находится на ранних стадиях, однако его глубокое влияние уже неоспоримо, что побуждает нас рассматривать как его огромные возможности, так и присущие ему проблемы.
인공지능의 빠른 발전은 우리가 일하고 소통하는 방식부터 배우고 창조하는 방식에 이르기까지 삶의 다양한 측면을 재편하고 있습니다. AI의 하위 분야인 기계 학습은 방대한 양의 데이터를 처리하고, 패턴을 식별하며, 놀라운 정확도로 예측을 수행할 수 있는 정교한 알고리즘 개발을 가능하게 했습니다. 이는 자연어 처리, 컴퓨터 비전, 로봇 공학 등 다양한 분야에서 획기적인 발전을 가져왔습니다. 예를 들어, 대규모 언어 모델(LLM)은 이제 일관성 있고 문맥상 적절한 텍스트를 생성하고, 인상적인 유창성으로 언어를 번역하며, 심지어 복잡한 문제 해결을 돕는 데에도 사용될 수 있습니다. AI의 윤리적 함의 또한 중요한 논의 영역입니다. 데이터 프라이버시, 알고리즘 편향, 고용에 미치는 잠재적 영향에 대한 질문은 연구자, 정책 입안자 및 일반 대중에 의해 활발하게 논의되고 있습니다. AI 개발이 공정성, 투명성 및 책임의 원칙에 따라 이루어지도록 보장하는 것은 사회적 이익을 위한 잠재력을 활용하는 데 매우 중요합니다. 의료, 금융, 제조와 같은 산업에 AI를 통합하는 것은 이미 상당한 효율성과 혁신을 가져오고 있습니다. 의료 분야에서 AI는 질병 진단, 치료 계획 개인화 및 신약 발견 가속화를 돕습니다. 금융 분야에서는 사기 탐지, 위험 관리 및 투자 전략 최적화에 기여합니다. AI의 미래는 설명 가능한 AI, 인공 일반 지능, 인간-AI 협업과 같은 분야에 초점을 맞춘 연구와 함께 더욱 혁신적인 변화를 약속합니다. 이러한 변화를 이해하고 적응하는 것은 AI 중심의 세상에서 개인과 조직 모두가 번성하는 데 핵심이 될 것입니다. AI 개발의 여정은 아직 초기 단계에 있지만, 그 심오한 영향은 이미 부인할 수 없으며, 우리는 그 엄청난 기회와 내재된 도전을 모두 고려해야 합니다."""
EMBEDDING_TEST_TEXT = os.getenv("EMBEDDING_TEST_TEXT", DEFAULT_EMBEDDING_TEST_TEXT_SHORT)
REPORT_FILE = "embedding_model_test_report.json"
REQUEST_TIMEOUT = 60  # seconds

# --- Helper Functions ---


def login_and_get_session() -> Optional[httpx.Client]:
    """Login to the system and return an authenticated httpx client."""
    try:
        client = httpx.Client(base_url=API_BASE_URL, timeout=REQUEST_TIMEOUT)

        # Login to get session cookies
        login_data = {"username": USERNAME, "password": PASSWORD}
        response = client.post("/api/v1/login", json=login_data)
        response.raise_for_status()

        print(f"Successfully logged in as {USERNAME}")
        return client

    except httpx.HTTPError as e:
        print(f"Login failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"Unexpected error during login: {e}")
        return None


def get_available_models(client: httpx.Client) -> Optional[Dict[str, Any]]:
    """Fetch all available models from the API."""
    try:
        print("Fetching available models...")
        # Get all models (not just recommended ones)
        request_data = {"tag_filters": []}
        response = client.post("/api/v1/available_models", json=request_data)
        response.raise_for_status()

        data = response.json()
        print("Successfully fetched models.")
        return data

    except httpx.HTTPError as e:
        print(f"Error fetching available models: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"Unexpected error fetching models: {e}")
        return None


def test_embedding_model(client: httpx.Client, provider: str, model: str, text: str) -> Dict[str, Any]:
    """Test a specific embedding model and return a result dictionary."""
    start_time = time.time()

    try:
        request_body = {"provider": provider, "model": model, "input": text}

        response = client.post("/api/v1/embeddings", json=request_body)
        response.raise_for_status()

        data = response.json()
        end_time = time.time()

        # Validate response structure
        if (
            "data" in data
            and len(data["data"]) > 0
            and "embedding" in data["data"][0]
            and isinstance(data["data"][0]["embedding"], list)
        ):
            dimension = len(data["data"][0]["embedding"])
            return {
                "test_pass": True,
                "dimension": dimension,
                "response_time_seconds": round(end_time - start_time, 2),
                "error_message": None,
            }
        else:
            return {
                "test_pass": False,
                "dimension": None,
                "response_time_seconds": round(end_time - start_time, 2),
                "error_message": "Invalid response format from API.",
            }

    except httpx.HTTPError as e:
        end_time = time.time()
        error_details = f"HTTP Error: {e.response.status_code}"
        try:
            # Try to get more specific error from response body
            error_body = e.response.json()
            if isinstance(error_body, dict) and "message" in error_body:
                error_details += f" - {error_body['message']}"
            else:
                error_details += f" - {error_body}"
        except (json.JSONDecodeError, AttributeError):
            error_details += f" - {e.response.text}"

        return {
            "test_pass": False,
            "dimension": None,
            "response_time_seconds": round(end_time - start_time, 2),
            "error_message": error_details,
        }

    except Exception as e:
        end_time = time.time()
        return {
            "test_pass": False,
            "dimension": None,
            "response_time_seconds": round(end_time - start_time, 2),
            "error_message": f"Unexpected error: {str(e)}",
        }


def extract_embedding_models(available_models_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract embedding models from the available models response."""
    embedding_models = []

    providers = available_models_data.get("items", [])
    for provider in providers:
        provider_name = provider.get("name", "")
        embedding_list = provider.get("embedding", [])

        if embedding_list:
            for model_info in embedding_list:
                if model_info and isinstance(model_info, dict):
                    model_name = model_info.get("model", "")
                    if model_name:
                        embedding_models.append({"provider": provider_name, "model": model_name})

    return embedding_models


def main():
    """Main function to run the embedding model test."""
    print("=" * 60)
    print("ApeRAG Embedding Model Test")
    print("=" * 60)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Test Text: {EMBEDDING_TEST_TEXT}")
    print(f"Report File: {REPORT_FILE}")
    print("=" * 60)

    # Login and get authenticated session
    client = login_and_get_session()
    if not client:
        print("\nFailed to login. Exiting.")
        return

    try:
        # Get available models
        available_models_data = get_available_models(client)
        if not available_models_data:
            print("\nCould not retrieve available models. Exiting.")
            return

        # Extract embedding models
        embedding_models = extract_embedding_models(available_models_data)

        if not embedding_models:
            print("\nNo embedding models found to test.")
            return

        print(f"\nFound {len(embedding_models)} embedding models to test:")
        for model_info in embedding_models:
            print(f"  - {model_info['provider']} / {model_info['model']}")
        print()

        # Test each embedding model
        report: List[Dict[str, Any]] = []

        for i, model_info in enumerate(embedding_models, 1):
            provider = model_info["provider"]
            model = model_info["model"]

            print(f"[{i}/{len(embedding_models)}] Testing: {provider} / {model}")

            result = test_embedding_model(client, provider, model, EMBEDDING_TEST_TEXT)

            report_entry = {
                "provider": provider,
                "model": model,
                "dimension": result["dimension"],
                "test_pass": result["test_pass"],
                "response_time_seconds": result["response_time_seconds"],
                "error_message": result["error_message"],
            }
            report.append(report_entry)

            # Print result
            status = "✅ PASSED" if result["test_pass"] else "❌ FAILED"
            print(f"  Status: {status}")
            if result["dimension"]:
                print(f"  Dimension: {result['dimension']}")
            if result["response_time_seconds"]:
                print(f"  Response Time: {result['response_time_seconds']}s")
            if not result["test_pass"] and result["error_message"]:
                print(f"  Error: {result['error_message']}")
            print("-" * 50)

        # Generate summary
        passed_count = sum(1 for entry in report if entry["test_pass"])
        failed_count = len(report) - passed_count

        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total Models Tested: {len(report)}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {failed_count}")
        print(f"Success Rate: {passed_count / len(report) * 100:.1f}%")

        # Save report to file
        try:
            with open(REPORT_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "test_summary": {
                            "total_models": len(report),
                            "passed": passed_count,
                            "failed": failed_count,
                            "success_rate": round(passed_count / len(report) * 100, 1),
                            "test_text": EMBEDDING_TEST_TEXT,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                        "results": report,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            print(f"\nReport saved to: {os.path.abspath(REPORT_FILE)}")

        except IOError as e:
            print(f"\nError saving report file: {e}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
