import wave
from enum import Enum
from functools import lru_cache
from io import BytesIO
from typing import BinaryIO, List, Optional, Union

import numpy as np
from cartesia import AsyncCartesia, Cartesia
from pydantic import BaseModel
from timeout_function_decorator import timeout

from utts.config import MAXHITS, TIMEOUT, get_settings
from utts.utils import convert_to_enum


class Model(str, Enum):
    """Available models for Cartesia TTS API."""

    # Base models
    SONIC_2 = "sonic-2"
    SONIC_TURBO = "sonic-turbo"
    SONIC = "sonic"

    # Specific snapshots
    SONIC_2_2025_04_16 = "sonic-2-2025-04-16"
    SONIC_2_2025_03_07 = "sonic-2-2025-03-07"
    SONIC_TURBO_2025_03_07 = "sonic-turbo-2025-03-07"
    SONIC_2024_12_12 = "sonic-2024-12-12"
    SONIC_2024_10_19 = "sonic-2024-10-19"


class Language(str, Enum):
    """Available languages for Cartesia TTS API."""

    ENGLISH = "en"
    FRENCH = "fr"
    GERMAN = "de"
    SPANISH = "es"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    JAPANESE = "ja"
    HINDI = "hi"
    ITALIAN = "it"
    KOREAN = "ko"
    DUTCH = "nl"
    POLISH = "pl"
    RUSSIAN = "ru"
    SWEDISH = "sv"
    TURKISH = "tr"


class Voice(str, Enum):
    SARAH = "694f9389-aac1-45b6-b726-9d9369183238"
    SARAH_CURIOUS = "794f9389-aac1-45b6-b726-9d9369183238"
    SOPHIE = "bf0a246a-8642-498a-9950-80c35e9276b5"
    SAVANNAH = "78ab82d5-25be-4f7d-82b3-7ad64e5b85b2"
    BROOKE = "6f84f4b8-58a2-430c-8c79-688dad597532"
    CALM_FRENCH_WOMAN = "a8a1eb38-5f15-4c1d-8722-7ac0f329727d"
    GRIFFIN = "c99d36f3-5ffd-4253-803a-535c1bc9c306"
    ZIA = "32b3f3c5-7171-46aa-abe7-b598964aa793"
    MATEO = "79743797-2087-422f-8dc7-86f9efca85f1"
    WISE_LADY = "c8605446-247c-4d39-acd4-8f4c28aa363c"
    ETHAN = "00967b2f-88a6-4a31-8153-110a92134b9f"
    NICO = "afa425cf-5489-4a09-8a3f-d3cb1f82150d"
    ADELE = "1ade29fc-6b82-4607-9e70-361720139b12"
    CORINNE = "0c8ed86e-6c64-40f0-b252-b773911de6bb"
    FLORENCE = "f6ff7c0c-e396-40a9-a70b-f7607edb6937"
    KATIE = "f786b574-daa5-4673-aa0c-cbe3e8534c02"
    ELLEN = "5c9e800f-2a92-4720-969b-99c4ab8fbc87"
    MADISON = "02fe5732-a072-4767-83e3-a91d41d274ca"
    DAVID = "da69d796-4603-4419-8a95-293bfc5679eb"
    AYUSH = "791d5162-d5eb-40f0-8189-f19db44611d8"
    RUPALI = "56e35e2d-6eb6-4226-ab8b-9776515a7094"
    LORI_SCARED = "fb78f09f-f998-4061-ad51-d71f90388f0e"
    LORI_SURPRISED = "c2da2a3e-b0d6-46bf-a09a-68562617a50a"
    LORI_CURIOUS = "ba0add52-783c-4ec0-8b9c-7a6b60f99d1c"
    LORI_HAPPY = "8843adfb-77d3-455a-86f9-de0651555ec6"
    LORI_ANGRY = "5cc54223-ec0c-4c50-87e9-b9947264e1f4"
    LORI = "57c63422-d911-4666-815b-0c332e4d7d6a"
    LORI_DISGUSTED = "414da90b-16b3-4e88-86f5-3c3945e8fa4b"
    LORI_SAD = "2d01710c-7c77-4cf1-b0d0-5902a25f6e17"
    STEVE_DISGUSTED = "f96dc0b1-7900-4894-a339-81fb46d515a7"
    STEVE_CURIOUS = "c1c65fc2-528a-4dde-a2c4-f822785c2704"
    STEVE_SCARED = "b1ce5126-4d08-42c3-adef-d3eb39e90c7a"
    STEVE_HAPPY = "adde00e9-c98f-42ae-a94d-fc9f92f11c76"
    STEVE = "9fb269e7-70fe-4cbe-aa3f-28bdb67e3e84"
    STEVE_SAD = "80713a53-e484-4f69-9852-7891096016ac"
    STEVE_ANGRY = "7c8ba972-4960-4c43-bea0-8178e2205696"
    STEVE_SURPRISED = "6fd4f468-0345-4f41-81d0-3f48ebc295e0"
    DAVID_ANGRY = "fd098a10-ba9e-445e-b144-be2a9f3dac02"
    DAVID_SAD = "c4e848dc-d4fd-4bc8-90ea-8525563ec0e5"
    DAVID_CURIOUS = "b08c966e-2146-4592-99eb-3171a714a43c"
    DAVID_SCARED = "a3a4fe2a-d402-41d1-be7d-28f71eda755f"
    DAVID_DISGUSTED = "9d2b4a7f-7ced-4fb8-b570-9ce21fb931c8"
    DAVID_HAPPY = "6b622a1d-906f-44af-b60c-7bef365bf124"
    DAVID_SURPRISED = "10d17ae0-8f64-472a-be00-f00a98c729e0"
    LUKE_CURIOUS = "8e14933d-ecd7-402b-9505-795130d69b35"
    LUKE = "7b2c0a2e-3dd3-4a44-b16b-26ecd8134279"
    LUKE_DISGUSTED = "79b8126f-c5d9-4a73-8585-ba5e1a077ed6"
    LUKE_SURPRISED = "725d43d6-1196-480e-bd87-728ae5eff9e1"
    LUKE_SCARED = "63426c82-a0c9-4f23-a175-50eb64c95ec1"
    LUKE_ANGRY = "61001bc6-9064-40a4-b8b2-29178e0fa558"
    LUKE_SAD = "5c7b66c2-3b58-464d-8a12-093410a269c5"
    LUKE_HAPPY = "3d79b1fd-daaa-439c-bff3-903dc18e7684"
    KENNETH_CURIOUS = "cf14fdcd-24a0-4d63-958a-c784f33d8e7c"
    KENNETH_SAD = "cb605424-d682-48e9-94db-34cc567cf1c6"
    KENNETH_SURPRISED = "abe7dee1-6051-43d3-9a9f-1ac1312497a7"
    KENNETH_ANGRY = "aa086107-101b-4182-a628-c51186d74166"
    KENNETH = "911b8b22-887f-4caf-bf87-85d834c08708"
    KENNETH_DISGUSTED = "876c39e1-9ecd-42cd-b0c1-8b3906f0be19"
    KENNETH_HAPPY = "83e45f18-fac4-40db-a43b-03257883b437"
    KENNETH_SCARED = "64875a07-f57e-4a70-b702-4e3fb25efeda"
    MADISON_SURPRISED = "a5def41e-2e73-433f-92f7-5f1d99fef05d"
    MADISON_CURIOUS = "98c87826-dba2-44f4-b123-4c7e3c8a2647"
    MADISON_HAPPY = "62305e79-9d39-4643-b003-5e0b096fe4f4"
    MADISON_DISGUSTED = "5993c2c9-5d59-403e-b459-946c8b302086"
    MADISON_SCARED = "30236d07-62d0-4c63-abf7-df46aa45e473"
    MADISON_SAD = "27c12970-3efb-4f39-a78a-2fbb7bddc941"
    MADISON_ANGRY = "134838f5-ce7e-4876-ac32-6367b99daf83"
    AADHYA = "f91ab3e6-5071-4e15-b016-cde6f2bcd222"
    OLIVIA = "f31cc6a7-c1e8-4764-980c-60a361443dd1"
    SAMANTHA_ANGRY = "04bfd756-4fd4-42c2-9ccf-37f647c5bf54"
    SAMANTHA_YELLING = "d3e03deb-5439-4203-add1-ca9a7501eaa7"
    SAMANTHA_SAD = "5e10a334-7fa5-46d4-a64b-5ae6185da3fd"
    SAMANTHA_HAPPY = "761afc95-bef5-44dd-aa07-d3c678912e43"
    TORI = "d7e54830-4754-4b17-952c-bcdb7e80a2fb"
    CARRIE = "4af7c703-f2a9-45dd-a7fd-724cf7efc371"
    CATHY = "031851ba-cc34-422d-bfdb-cdbb7f4651ee"
    TALL_MAN = "e49cf445-3d04-486b-9acd-41fa7198c745"
    JOAN_OF_ARK = "c9440d34-5641-427b-bbb7-80ef7462576d"
    CAMILLE = "55deba52-bc73-4481-ab69-9c8831c8a7c3"
    BO = "d3b22900-ec95-4344-a548-2d34e9b842b7"
    CHEN = "7a5d4663-88ae-47b7-808e-8f9b9ee4127b"
    ISABEL = "c0c374aa-09be-42d9-9828-4d2d7df86962"
    CHONGZ = "146485fd-8736-41c7-88a8-7cdd0da34d84"
    KEITH = "9fa83ce3-c3a8-4523-accc-173904582ced"
    RONALD = "5ee9feff-1265-424a-9d7f-8e4d431a12c7"
    LIV = "d718e944-b313-4998-b011-d1cc078d4ef3"
    SQUEAKY_MATT = "1cf55ffc-5904-4483-9337-298042eab1db"
    GOOFY_MATT = "bfd3644b-d561-4b1c-a01f-d9af98cb67c0"
    NIGHT_WARDEN = "6a176356-ada1-4b48-b2ae-3a3fdd485680"
    LUCA_ENGLISH = "1fc31370-81b1-4588-9c1a-f93793c6e01d"
    LUCA = "e019ed7e-6079-4467-bc7f-b599a5dccf6f"
    LISA = "c378e743-e7dc-49da-b9ce-8377b543acdd"
    CHINESE_LISA = "bf32f849-7bc9-4b91-8c62-954588efcc30"
    JOAN = "5abd2130-146a-41b1-bcdb-974ea8e19f56"
    CONNIE = "8d8ce8c9-44a4-46c4-b10f-9a927b99a853"
    SAMANTHA = "f4e8781b-a420-4080-81cf-576331238efa"
    MARIO = "5ef98b2a-68d2-4a35-ac52-632a2d288ea6"
    JUAN = "b042270c-d46f-4d4f-8fb0-7dd7c5fe5615"
    DALLAS = "23e9e50a-4ea2-447b-b589-df90dbb848a2"
    COREY = "58db94c7-8a77-46a7-9107-b8b957f164a0"
    REBECCA = "57b6bf63-c7a1-4ffc-8e10-23bf45152dd6"
    CASPER = "4f7f1324-1853-48a6-b294-4e78e8036a83"
    DAVE = "ab109683-f31f-40d7-b264-9ec3e26fb85e"
    SALLY = "6adbb439-0865-468c-9e68-adbb0eb2e71c"
    STACY = "6d287143-8db3-434a-959c-df147192da27"
    BRENDA = "607167f6-9bf2-473c-accc-ac7b3b66b30b"
    JORDAN = "87bc56aa-ab01-4baa-9071-77d497064686"
    NATHAN = "97f4b8fb-f2fe-444b-bb9a-c109783a857a"
    PLEASANT_MAN = "729651dc-c6c3-4ee5-97fa-350da1f88600"
    HELPFUL_WOMAN = "156fb8d2-335b-4950-9cb3-a2d33befec77"
    SOUTHERN_WOMAN = "f9836c6e-a0bd-460e-9d3c-f7299fa60f94"
    FRIENDLY_SIDEKICK = "e00d0e4c-a5c8-443f-a8a3-473eb9a62355"
    MADAME_MISCHIEF = "e13cae5c-ec59-4f71-b0a6-266df3c9bb8e"
    AMERICAN_VOICEOVER_MAN = "7fe6faca-172f-4fd9-a193-25642b8fdb07"
    CUSTOMER_SERVICE_MAN = "2a4d065a-ac91-4203-a015-eb3fc3ee3365"
    AMERICAN_NARRATOR_LADY = "a8136a0c-9642-497a-882d-8d591bdcb2fa"
    AMERICAN_AD_READ_MAN = "64462aed-aafc-45d4-84cd-ecb4b3763a0a"
    HELP_DESK_MAN = "39b376fc-488e-4d0c-8b37-e00b72059fdd"
    HELP_DESK_WOMAN = "af346552-54bf-4c2b-a4d4-9d2820f51b6c"
    THE_MERCHANT = "50d6beb4-80ea-4802-8387-6c948fe84208"
    GRIFFIN_EXCITED = "34d923aa-c3b5-4f21-aac7-2c1f12730d4b"
    OLD_TIMEY_RADIO_MAN = "236bb1fb-dc41-4a2b-84d6-d22d2a2aaae1"
    OVERLORD = "224126de-034c-429b-9fde-71031fba9a59"
    ROBOTIC_MALE = "185c2177-de10-4848-9c0a-ae6315ac1493"
    HEROIC_MAN = "ec58877e-44ae-4581-9078-a04225d42bd4"
    THE_ORACLE = "d7862948-75c3-4c7c-ae28-2959fe166f49"
    AUSTRALIAN_PROMOTER_MAN = "a3afd376-04f9-48e2-a966-132cdfdbc093"
    AUSTRALIAN_SALESMAN = "da4a4eff-3b7e-4846-8f70-f075ff61222c"
    AUSTRALIAN_SUPPORT_AGENT = "34b3e510-dd50-4a8d-86d7-478e7ee5a9e8"
    YIPPY = "8f490c09-3b8f-4108-ac0c-51288154fa05"
    ARBUCKLE = "572339a6-ba03-4d07-ac2a-0b86308d1ea2"
    BRIGHTON = "7447a397-30c1-4681-b687-0ed1b7abf0fb"
    CLARION = "8d110413-2f14-44a2-8203-2104db4340e9"
    SILAS = "7e19344f-9f17-47d7-a13a-4366ad06ebf3"
    ORION = "701a96e1-7fdd-4a6c-a81e-a4a450403599"
    GRANT = "63406bbd-ce1b-4fff-8beb-86d3da9891b9"
    LENA = "4ab1ff51-476d-42bb-8019-4d315f7c0c05"
    LITTLE_GAMING_GIRL = "cccc21e8-5bcf-4ff0-bc7f-be4e40afc544"
    LITTLE_NARRATOR_GIRL = "56b87df1-594d-4135-992c-1112bb504c59"
    BENEDICT = "7cf0e2b1-8daf-4fe4-89ad-f6039398f359"
    GREGOR = "e2569545-f8d1-4c24-bfaf-73f951052337"
    POSITIVE_SHY_MAN = "c2488032-7cba-449c-9036-9e11b69286a1"
    DORIAN = "586b6832-1ca1-43ad-b974-527dc13c2532"
    COMMERCIAL_LADY = "c2ac25f9-ecc4-4f56-9095-651354df60c0"
    NEWSMAN = "d46abd1d-2d02-43e8-819f-51fb652c1c61"
    HARRY = "3dcaa773-fb1a-47f7-82a4-1bf756c4e1fb"
    LENNY = "4629672e-661d-4f59-93fc-8db4476b585f"
    CLASSY_BRITISH_MAN = "95856005-0332-41b0-935f-352e296aa0df"
    ELENA = "cefcb124-080b-4655-b31f-932f3ee743de"
    LUCIO = "e5923af7-a329-4e9b-b95a-5ace4a083535"
    ALINA = "38aabb6a-f52b-4fb0-a3d1-988518f4dc06"
    LUKAS = "e00dd3df-19e7-4cd4-827a-7ff6687b6954"
    WIZARDMAN = "87748186-23bb-4158-a1eb-332911b0b708"
    CUSTOMER_SUPPORT_MAN = "a167e0f3-df7e-4d52-a9c3-f949145efdab"
    NONFICTION_MAN = "79f8b5fb-2cc8-479a-80df-29f7a7cf1a3e"
    CALM_LADY = "00a77add-48d5-4ef6-8157-71e5437b282d"
    CHINESE_READING_WOMAN = "f9a4b3a6-b44b-469f-90e3-c8e19bd30e99"
    CHINESE_LECTURER_MAN = "c59c247b-6aa9-4ab6-91f9-9eabea7dc69e"
    TURKISH_NARRATOR_LADY = "bb2347fe-69e9-4810-873f-ffd759fe8420"
    CASTILIAN_SPANISH_PRESENTER_WOMAN = "d4db5fb9-f44b-4bd1-85fa-192e0f0d75f9"
    CASTILIAN_SPANISH_PRESENTER_MAN = "b5aa8098-49ef-475d-89b0-c9262ecf33fd"
    CASUAL_BRAZILIAN_MAN = "a37639f0-2f0a-4de4-9942-875a187af878"
    CONVERSATIONAL_BRAZILIAN_WOMAN = "1cf751f6-8749-43ab-98bd-230dd633abdb"
    KOREAN_SUPPORT_WOMAN = "304fdbd8-65e6-40d6-ab78-f9d18b9efdf9"
    KOREAN_NARRATOR_MAN = "af6beeea-d732-40b6-8292-73af0035b740"
    TAYLAN = "c1cfee3d-532d-47f8-8dd2-8e5b2b66bf1d"
    LEYLA = "fa7bfcdc-603c-4bf1-a600-a371400d2f8c"
    LARS = "0caedb75-417f-4e36-9b64-c21354cb94c8"
    FREJA = "6c6b05bf-ae5f-4013-82ab-7348e99ffdb2"
    TATIANA = "064b17af-d36b-4bfb-b003-be07dba1b649"
    JAKUB = "2a3503b2-b6b6-4534-a224-e8c0679cec4a"
    ZOFIA = "dcf62f33-7cff-4f20-85b2-2efaa68cbc32"
    KENJI = "6b92f628-be90-497c-8f4c-3b035002df71"
    YUKI = "59d4fd2f-f5eb-4410-8105-58db7661144f"
    MARCO = "79693aee-1207-4771-a01e-20c393c89e6f"
    FRANCESCA = "d609f27f-f1a4-410f-85bb-10037b4fba99"
    AMIT = "9b953e7b-86a8-42f0-b625-1434fb15392b"
    SEBASTIAN = "b7187e84-fe22-4344-ba4a-bc013fcb533e"
    LUCAS = "af482421-80f4-4379-b00c-a118def29cde"
    SANNE = "0eb213fe-4658-45bc-9442-33a48b24b133"
    TIAGO = "6a360542-a117-4ed5-9e09-e8bf9b05eabb"
    CLARA = "d4b44b9a-82bc-4b65-b456-763fce4c52f9"
    MIA = "1d3ba41a-96e6-44ad-aabb-9817c56caa68"
    GRACE = "a38e4e85-e815-43ab-acf1-907c4688dd6c"
    JOHN = "f785af04-229c-4a7c-b71b-f3194c7f08bb"
    REN_THE_FURY = "9e7ef2cf-b69c-46ac-9e35-bbfd73ba82af"
    LILY_WHISPER = "c7eafe22-8b71-40cd-850b-c5a3bbd8f8d2"
    COMMANDING_JAPANESE_MAN = "446f922f-c43a-4aad-9a8b-ad2af568e882"
    YOUNG_SHY_JAPANESE_WOMAN = "0cd0cde2-3b93-42b5-bcb9-f214a591aa29"
    INTENSE_JAPANESE_MAN = "a759ecc5-ac21-487e-88c7-288bdfe76999"
    TAKESHI = "06950fa3-534d-46b3-93bb-f852770ea0b5"
    RUSSIAN_STORYTELLER_MAN = "da05e96d-ca10-4220-9042-d8acef654fa9"
    ANU = "87177869-f798-48ae-870f-e07d0c960a1e"
    JANVI = "7ea5e9c2-b719-4dc3-b870-5ba5f14d31d8"
    PRIYA = "f6141af3-5f94-418c-80ed-a45d450e7e2e"
    KIARA = "f8f5f1b2-f02d-4d8e-a40d-fd850a487b3d"
    ADITI = "1998363b-e108-4736-bc5b-1449fa2b096a"
    DEVANSH = "1259b7e3-cb8a-43df-9446-30971a46b8b0"
    NEIL = "a0cc0d65-5317-4652-b166-d9d34a244c6f"
    INDIAN_CONVERSATIONAL_WOMAN = "9cebb910-d4b7-4a4a-85a4-12c79137724c"
    APOORVA = "faf0731e-dfb9-4cfc-8119-259a79b27e12"
    ANANYA = "28ca2041-5dda-42df-8123-f58ea9c3da00"
    ISHAN = "fd2ada67-c2d9-4afe-b474-6386b87d8fc3"
    PARVATI = "bec003e2-3cb3-429c-8468-206a393c67ad"
    MIHIR = "be79f378-47fe-4f9c-b92b-f02cefa62ccf"
    HINDI_REPORTER_MAN = "bdab08ad-4137-4548-b9db-6142854c7525"
    DUTCH_CONFIDENT_MAN = "9e8db62d-056f-47f3-b3b6-1b05767f9176"
    KOREAN_NARRATOR_WOMAN = "663afeec-d082-4ab5-827e-2e41bf73a25b"
    RUSSIAN_NARRATOR_WOMAN = "642014de-c0e3-4133-adc0-36b5309c23e6"
    TURKISH_NARRATOR_MAN = "5a31e4fb-f823-4359-aa91-82c0ae9a991c"
    POLISH_NARRATOR_WOMAN = "575a5d29-1fdc-4d4e-9afa-5a9a71759864"
    POLISH_NARRATOR_MAN = "4ef93bb3-682a-46e6-b881-8e157b6b4388"
    DUTCH_MAN = "4aa74047-d005-4463-ba2e-a0d9b261fb87"
    ITALIAN_CALM_MAN = "408daed0-c597-4c27-aae8-fa0497d644bf"
    POLISH_CONFIDENT_MAN = "3d335974-4c4a-400a-84dc-ebf4b73aada6"
    TURKISH_CALM_MAN = "39f753ef-b0eb-41cd-aa53-2f3c284f948f"
    SWEDISH_NARRATOR_MAN = "38a146c3-69d7-40ad-aada-76d5a2621758"
    RUSSIAN_NARRATOR_MAN = "2b3bb17d-26b9-421f-b8ca-1dd92332279f"
    KOREAN_CALM_WOMAN = "29e5f8b4-b953-4160-848f-40fae182235b"
    ITALIAN_NARRATOR_WOMAN = "0e21713a-5e9a-428a-bed4-90d410b87f13"
    ITALIAN_NARRATOR_MAN = "029c3c7a-b6d9-44f0-814b-200d849830ff"
    FRIENDLY_AUSTRALIAN_MAN = "421b3369-f63f-4b03-8980-37a44df1d4e8"
    AUSTRALIAN_NARRATOR_LADY = "8985388c-1332-4ce7-8d55-789628aa3df4"
    AUSTRALIAN_CUSTOMER_SUPPORT_MAN = "41f3c367-e0a8-4a85-89e0-c27bae9c9b6d"
    MEXICAN_WOMAN = "5c5ad5e7-1020-476b-8b91-fdcbe9cc313c"
    MEXICAN_MAN = "15d0c2e2-8d29-44c3-be23-d585d5f154a1"
    POLITE_MAN = "ee7ea9f8-c0c1-498c-9279-764d6b56d189"
    CALIFORNIA_GIRL = "b7d50908-b17c-442d-ad8d-810c63997ed9"
    GERMAN_WOMAN = "b9de4a89-2257-424b-94c2-db18ba68c81a"
    FRIENDLY_BRAZILIAN_MAN = "6a16c1f4-462b-44de-998d-ccdaa4125a0a"
    GERMAN_CONVERSATION_MAN = "384b625b-da5d-49e8-a76d-a2855d4f31eb"
    GERMAN_STORYTELLER_MAN = "db229dfe-f5de-4be4-91fd-7b077c158578"
    STERN_FRENCH_MAN = "0418348a-0ca2-4e90-9986-800fb8b3bbc0"
    FRENCH_NARRATOR_MAN = "5c3c89e5-535f-43ef-b14d-f8ffe148c1f0"
    FRENCH_NARRATOR_LADY = "8832a0b5-47b2-4751-bb22-6a8e2149303d"
    SPANISH_SPEAKING_REPORTER_MAN = "2695b6b5-5543-4be1-96d9-3967fb5e7fec"
    SPANISH_STORYTELLER_MAN = "846fa30b-6e1a-49b9-b7df-6be47092a09a"
    STORYTELLER_LADY = "996a8b96-4804-46f0-8e05-3fd4ef1a87cd"
    ASMR_LADY = "03496517-369a-4db1-8236-3d3ae459ddf7"
    TUTORIAL_MAN = "bd9120b6-7761-47a6-a446-77ca49132781"
    TEACHER_LADY = "573e3144-a684-4e72-ac2b-9b2063a50b53"
    COMMERCIAL_MAN = "7360f116-6306-4e9a-b487-1235f35a0f21"
    BRITISH_CUSTOMER_SUPPORT_LADY = "a01c369f-6d2d-4185-bc20-b32c225eab70"
    PRINCESS = "8f091740-3df1-4795-8bd9-dc62d88e5131"
    AUSTRALIAN_WOMAN = "043cfc81-d69f-4bee-ae1e-7862cb358650"
    AUSTRALIAN_MAN = "13524ffb-a918-499a-ae97-c98c7c4408c4"
    CUSTOMER_SUPPORT_LADY = "829ccd10-f8b3-43cd-b8a0-4aeaa81f3b30"
    SALESMAN = "820a3788-2b37-4d21-847a-b65d8a68c99a"
    FEMALE_NURSE = "5c42302c-194b-4d0c-ba1a-8cb485c84ab9"
    LAIDBACK_WOMAN = "21b81c14-f85b-436d-aff5-43f2e788ecf8"
    KENTUCKY_WOMAN = "4f8651b0-bbbd-46ac-8b37-5168c5923303"
    KENTUCKY_MAN = "726d5ae5-055f-4c3d-8355-d9677de68937"
    MIDWESTERN_WOMAN = "11af83e2-23eb-452f-956e-7fee218ccb5c"
    WISE_GUIDE_MAN = "42b39f37-515f-4eee-8546-73e841679c1d"
    READING_MAN = "f146dcec-e481-45be-8ad2-96e1e40e7f32"
    ANNOUNCER_MAN = "5619d38c-cf51-4d8e-9575-48f61a280413"
    CHINESE_COMMERCIAL_MAN = "eda5bbff-1ff1-4886-8ef1-4e69a77640a0"
    CHINESE_FEMALE_CONVERSATIONAL = "e90c6678-f0d3-4767-9883-5d0ecf5894a8"
    CHINESE_WOMAN_NARRATOR = "d4d4b115-57a0-48ea-9a1a-9898966c2966"
    FRIENDLY_FRENCH_MAN = "ab7c61f5-3daa-47dd-a23b-4ac0aac5f5c3"
    PLEASANT_BRAZILIAN_LADY = "700d1ee3-a641-4018-ba6e-899dcadc9e2b"
    HELPFUL_FRENCH_LADY = "65b25c5d-ff07-4687-a04c-da2f43ef6fa9"
    BRAZILIAN_YOUNG_MAN = "5063f45b-d9e0-4095-b056-8f3ee055d411"
    JAPANESE_CHILDREN_BOOK = "44863732-e415-4084-8ba1-deabe34ce3d2"
    GERMAN_CONVERSATIONAL_WOMAN = "3f4ade23-6eb4-4279-ab05-6a144947c4d5"
    CHINESE_COMMERCIAL_WOMAN = "0b904166-a29f-4d2e-bb20-41ca302f98e9"
    YOGAMAN = "f114a467-c40a-4db8-964d-aaba89cd08fa"
    BARBERSHOP_MAN = "a0e99841-438c-4a64-b679-ae501e7d6091"
    SPORTSMAN = "ed81fd13-2016-4a49-8fe3-c0d2761695fc"
    MOVIEMAN = "c45bc5ec-dc68-4feb-8829-6e6b2748095d"
    NEWSLADY = "bf991597-6c13-47e4-8411-91ec2de5c466"
    PROFESSIONAL_WOMAN = "248be419-c632-4f23-adf1-5324ed7dbf1d"
    CONFIDENT_BRITISH_MAN = "63ff761f-c1e8-414b-b969-d1833d1c870c"
    SOUTHERN_MAN = "98a34ef2-2140-4c28-9c71-663dc4dd7022"
    INDIAN_LADY = "3b554273-4299-48b9-9aaf-eefd438e3941"
    PILOT_OVER_INTERCOM = "36b42fcb-60c5-4bec-b077-cb1a00a92ec6"
    MIDDLE_EASTERN_WOMAN = "daf747c6-6bc2-4083-bd59-aa94dce23f5d"
    SWEET_LADY = "e3827ec5-697a-4b7c-9704-1a23041bbc51"
    INDIAN_MAN = "638efaaa-4d0c-442e-b701-3fae16aad012"
    JAPANESE_WOMAN_CONVERSATIONAL = "2b568345-1d48-4047-b25f-7baccf842eb0"
    READING_LADY = "15a9cd88-84b0-4a8b-95f2-5d583b54c72e"
    MIDWESTERN_MAN = "565510e8-6b45-45de-8758-13588fbaec73"
    ALABAMA_MAN = "40104aff-a015-4da1-9912-af950fbec99e"
    NEW_YORK_WOMAN = "34bde396-9fde-4ebf-ad03-e3a1d1155205"
    NEW_YORK_MAN = "34575e71-908f-4ab6-ab54-b08c95d6597d"
    DOCTOR_MISCHIEF = "fb26447f-308b-471e-8b00-8e9f04284eb5"
    BRITISH_READING_LADY = "71a7ad14-091c-4e8e-a314-022ece01c121"
    JAPANESE_MALE_CONVERSATIONAL = "e8a863c6-22c7-4671-86ca-91cacffc038d"
    ANNA = "91b4cf29-5166-44eb-8054-30d40ecc8081"
    RUSSIAN_CALM_WOMAN = "779673f3-895f-4935-b6b5-b031dc78b319"
    HINGLISH_SPEAKING_WOMAN = "95d51f79-c397-46f9-b49a-23763d3eaa2d"
    POLISH_YOUNG_MAN = "82a7fc13-2927-4e42-9b8a-bb1f9e506521"
    SWEDISH_CALM_LADY = "f852eb8d-a177-48cd-bf63-7e4dcab61a36"
    JAPANESE_NARRATION_MAN = "97e7d7a9-dfaa-4758-a936-f5f844ac34cc"
    HINDI_NARRATOR_MAN = "7f423809-0011-4658-ba48-a411f5e516ba"
    SABINE = "11c61307-4f9e-4db8-ac3b-bfa5f2a731ce"
    JACQUELINE = "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"


@lru_cache(MAXHITS)
def get_client() -> Cartesia:
    """Returns a Cartesia client."""

    settings = get_settings().cartesia
    assert settings is not None, "Cartesia settings are not configured"

    return Cartesia(api_key=settings.api_key)


@lru_cache(MAXHITS)
def get_async_client() -> AsyncCartesia:
    """Returns an async Cartesia client."""

    settings = get_settings().cartesia
    assert settings is not None, "Cartesia settings are not configured"

    return AsyncCartesia(api_key=settings.api_key)


@timeout(TIMEOUT)
def generate(
    text: str,
    model: Union[Model, str] = Model.SONIC_2,
    language: Union[Language, str] = Language.ENGLISH,
    voice: Union[Voice, str] = Voice.SARAH,
    voice_audio: Optional[Union[bytes, BinaryIO]] = None,
    duration: Optional[float] = None,
) -> bytes:
    """
    Generates audio from text using Cartesia TTS API.

    Args:
        text: Text to convert to speech
        model: TTS model to use (sonic-2, sonic-turbo, sonic)
        language: Language code (en, fr, de, etc.)
        voice: Voice to use (enum or ID string)
        voice_audio: Audio sample for voice cloning
        duration: Target duration in seconds for the generated audio

    Returns:
        Audio data as bytes
    """
    client = get_client()
    model_enum = convert_to_enum(Model, model)
    language_enum = convert_to_enum(Language, language)

    params = {
        "model_id": model_enum.value,
        "transcript": text,
        "language": language_enum.value,
        "output_format": {
            "container": "wav",
            "sample_rate": 44100,
            "encoding": "pcm_f32le",
        },
    }

    # Add voice - required parameter
    if voice_audio:
        params["voice"] = {"audio": voice_audio}
    else:
        # Use voice enum or string
        voice_enum = convert_to_enum(Voice, voice)
        params["voice"] = {"id": voice_enum.value}

    # Add duration parameter if specified
    if duration is not None:
        params["duration"] = duration

    # Collect bytes from iterator
    output = BytesIO()
    for chunk in client.tts.bytes(**params):
        output.write(chunk)

    return output.getvalue()


@timeout(TIMEOUT)
async def agenerate(
    text: str,
    model: Union[Model, str] = Model.SONIC_2,
    language: Union[Language, str] = Language.ENGLISH,
    voice: Union[Voice, str] = Voice.SARAH,
    voice_audio: Optional[Union[bytes, BinaryIO]] = None,
    duration: Optional[float] = None,
) -> bytes:
    """
    Asynchronously generates audio from text using Cartesia TTS API.

    Args:
        text: Text to convert to speech
        model: TTS model to use (sonic-2, sonic-turbo, sonic)
        language: Language code (en, fr, de, etc.)
        voice: Voice to use (enum or ID string)
        voice_audio: Audio sample for voice cloning
        duration: Target duration in seconds for the generated audio

    Returns:
        Audio data as bytes
    """
    client = get_async_client()
    model_enum = convert_to_enum(Model, model)
    language_enum = convert_to_enum(Language, language)

    params = {
        "model_id": model_enum.value,
        "transcript": text,
        "language": language_enum.value,
        "output_format": {
            "container": "wav",
            "sample_rate": 44100,
            "encoding": "pcm_f32le",
        },
    }

    # Add voice - required parameter
    if voice_audio:
        params["voice"] = {"audio": voice_audio}
    else:
        # Use voice enum or string
        voice_enum = convert_to_enum(Voice, voice)
        params["voice"] = {"id": voice_enum.value}

    # Add duration parameter if specified
    if duration is not None:
        params["duration"] = duration

    # Collect bytes from async iterator
    output = BytesIO()
    async for chunk in client.tts.bytes(**params):
        output.write(chunk)

    return output.getvalue()


def _wrap_pcm_f32_to_wav(raw_data: bytes, sample_rate: int = 44100) -> bytes:
    """
    Convert raw PCM float32 data to WAV with PCM S16LE encoding.
    """
    # Interpret raw data as float32
    floats = np.frombuffer(raw_data, dtype=np.float32)
    # Scale to int16 range
    ints = np.clip(floats * 32767.0, -32768, 32767).astype(np.int16)
    # Write WAV
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for PCM 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(ints.tobytes())
    return buf.getvalue()


class TTSWithTimestampsResponse(BaseModel):
    """
    Pydantic model for TTS response with word and phoneme timestamps.
    """

    audio: bytes  # WAV bytes
    words: List[str]
    word_starts: List[float]
    word_ends: List[float]
    phonemes: List[str]
    phoneme_starts: List[float]
    phoneme_ends: List[float]


@timeout(TIMEOUT)
async def agenerate_with_timestamps(
    text: str,
    model: Union[Model, str] = Model.SONIC_2,
    language: Union[Language, str] = Language.ENGLISH,
    voice: Union[Voice, str] = Voice.SARAH,
    voice_audio: Optional[Union[bytes, BinaryIO]] = None,
) -> TTSWithTimestampsResponse:
    """
    Asynchronously generates audio and returns a Pydantic object with WAV bytes,
    word-level and phoneme-level timestamps.

    Args:
        text: Text to convert to speech
        model: TTS model to use (sonic-2, sonic-turbo, sonic)
        language: Language code (en, fr, de, etc.)
        voice: Voice to use (enum or ID string)
        voice_audio: Audio sample for voice cloning
    """
    client = get_async_client()
    model_enum = convert_to_enum(Model, model)
    language_enum = convert_to_enum(Language, language)
    voice_enum = convert_to_enum(Voice, voice)

    ws = await client.tts.websocket()
    params = {
        "model_id": model_enum.value,
        "transcript": text,
        "language": language_enum.value,
        "voice": {"audio": voice_audio} if voice_audio else {"id": voice_enum.value},
        "output_format": {"container": "raw", "encoding": "pcm_f32le", "sample_rate": 44100},
        "add_timestamps": True,
        "add_phoneme_timestamps": True,
        "stream": True,
    }

    stream = await ws.send(**params)
    audio_chunks: List[bytes] = []
    words: List[str] = []
    word_starts: List[float] = []
    word_ends: List[float] = []
    phonemes: List[str] = []
    phoneme_starts: List[float] = []
    phoneme_ends: List[float] = []

    async for out in stream:  # type: ignore
        if out.audio:
            audio_chunks.append(out.audio)
        if out.word_timestamps:
            words.extend(out.word_timestamps.words)
            word_starts.extend(out.word_timestamps.start)
            word_ends.extend(out.word_timestamps.end)
        if hasattr(out, "phoneme_timestamps") and out.phoneme_timestamps:
            phonemes.extend(out.phoneme_timestamps.phonemes)
            phoneme_starts.extend(out.phoneme_timestamps.start)
            phoneme_ends.extend(out.phoneme_timestamps.end)

    await ws.close()
    raw = b"".join(audio_chunks)
    wav = _wrap_pcm_f32_to_wav(raw)

    return TTSWithTimestampsResponse(
        audio=wav,
        words=words,
        word_starts=word_starts,
        word_ends=word_ends,
        phonemes=phonemes,
        phoneme_starts=phoneme_starts,
        phoneme_ends=phoneme_ends,
    )
