# Rumor Verification using Evidence from Authorities
Given a rumor expressed in a tweet and a set of authorities (one or more authority Twitter accounts) for that rumor,
represented by a list of tweets from their timelines during the period surrounding the rumor,the system should retrieve up to
5 evidence tweets from those timelines, and determine if the rumor is supported (true), refuted (false), or unverifiable (in 
case not enough evidence to verify it exists in the given tweets) according to the evidence. This task is offered in both
Arabic and English.

__Table of Content:__

- [Input Data Format](#input-data-format)
- [Output Data Format](#output-data-format)

# Proposed Approach (Training model)
```mermaid
graph LR
A[Load data] --> B(Preprocess
        - Noise removal
        - URLs removal)
    B --> C(Extract features
        - SBERT embeddings)
    C --> D(Identify relevant tweets
        - Cosine similarity between rumor and evidence
        - Determine threshold using xx
        - Add tolerance) & E (- Use augementation with with relevant tweets *optional*
                              - Generate synthetic data *optional*)

    D --> F(Fine-tune pretrained stance detection model
        - Use pretrained model xx
        - Fine-tune with evidence: more weight
        - Fine-tune with relevant samples from timeline: lesser weight) & E --> F
    F --> G[Determine threshold
        - Identify probability of of relevant instance for the chosen label
        - Arrange in decreasing order of this score
        - Use cutoff *determined from rumor/evidence distance*] 
```


# Input Data Format

We provide train and dev data in JSON format files. Each file contains a list of JSON objects representing rumors. For each rumor, we provide the following entries:
```
{
  id [unique ID for the rumor]
  rumor [rumor tweet text]
  label [the veracity label of the rumor either SUPPORTS, REFUTES, NOT ENOUGH INFO]
  timeline [authorities timeline associated with the rumor each authority tweet is represented by authority Twitter account link, authority tweet ID, authority tweet text]
  evidence [authorities evidence tweets represented by authority Twitter account link, authority tweet ID, authority tweet text]
}
```
**Examples for Arabic data**:

```
{
  "id": "AuRED_089",
  "rumor": "وباء كورونا وصل الى الامارات 75 إصابة في ابوظبي و 63 إصابة في دبي  تحذير للامتناع عن السفر الى الامارات حفاظًا على السلامه و عدم نقل الوباء . اللهم أحفظ المسلمين في كل مكان..." ,
  "label": "REFUTES"
  "timeline": [["https://twitter.com/WHOEMRO", "1222971333522468867", "منظمة الصحة العالمية تعلن فاشية #فيروس_كورونا المستجد طارئة صحة عامة تثير قلقاً دوليا https://t.co/pVOXpZaPH7"],
   ["https://twitter.com/WHOEMRO", "1223608938136047616", "س. هل تحمي اللقاحات المضادة للالتهاب الرئوي من #فيروس_كورونا المستجد؟ ج. لا. لقاحات الالتهاب الرئوي لا تحمي من فيروس كورونا المستجد. هذا الفيروس جديد ومختلف ويحتاج لقاحاً خاصاً به. الباحثون يعملون على تطوير لقاح مضاد لهذا الفيروس. #اعرف_الحقائق https://t.co/QTGmI2flo9"],
   ["https://twitter.com/mohapuae", "1223361274618183681", "تعرف على أعراض فيروس كورونا الجديد #فيروس_كورونا_الجديد #فيروس_كورونا#كورونا#وزارة_الصحة_ووقاية_المجتمع_الإمارات https://t.co/jWALFtA68m"],
   ["https://twitter.com/mohapuae", "1223279618372882432", "مقتطفات من مشاركة وزارة الصحة ووقاية المجتمع في معرض ومؤتمر الصحة العربي2020 من خلال مجموعة من مبادرات ومشاريع الرعاية الصحية المبتكرة تحت شعار "صحة الإمارات مسؤولية مشتركة"#وزارة_الصحة_ووقاية_المجتمع_الإمارات#معرض_ومؤتمر_الصحة_العربي_2020#صحة_الإمارات https://t.co/c69pHj6ffd"],
   ......],
  "evidence": [["https://twitter.com/WHOEMRO","1222506828694794240","أكدت اليوم @WHO ظهور أولى حالات فيروس كورونا المستجد في إقليم شرق المتوسط، بالإمارات العربية المتحدة. عقب تأكيد @mohapuae في 29 يناير.
كان 4 أفراد من نفس العائلة من مدينة ووهان الصينية وصلوا إلى الإمارات في بداية يناير 2020، وتم إدخالهم المستشفى بعد تأكد إصابتهم ب #فيروس_كورونا."],
   ["https://twitter.com/mohapuae", "1222476311291142145", "إصابة أربعة أشخاص من عائلة صينية بفيروس كورونا الجديد جميعهم في حالة مستقرة وتم احتواؤهم وفق الإجراءات الاحترازية المعتمدة عالميا#وزارة_الصحة_ووقاية_المجتمع_الإمارات #فيروس_كورونا_الجديد #فيروس_كورونا https://t.co/ydy2esb20B"]
  ,....]
},
...,
{
  "id": "AuRED_105",
  "rumor": "تونس تعرض مساعدة ليبيا في علاج مصابي انفجار شاحنة الوقود في بنت بيّة #ليبيا #الشاهد للتفاصيل: https://t.co/s7fdU5fvgq" ,
  "label": "SUPPORTS",
  "timeline": [["https://twitter.com/NajlaElmangoush", "1554448728320344064", "أتقدم بالشكر لفخامة رئيس جمهورية #تونس السيد قيس سعيد @TnPresidencyعلى تضامنه وتسخير كل المستشفيات والأطقم الطبية لمساعدة جرحى #بنت_بيه وهذا التضامن يدل على أن ما يجمع الشعبين الشقيقين هو علاقات أخوية وروح تضامنية في السراء والضراء في كل الحالات @OJerandi 🇱🇾🇹🇳"],
  ["https://twitter.com/NajlaElmangoush", "1554027191788355584", "استفاقت بلدية #بنت_بية فجر اليوم على كارثة إنسانية وخبر مفزع، نتيجة انفجار صهريج الوقود، أسفر عن وفاة 5 أشخاص وإصابة قرابة 50 أخرين، أقدم تعازينا الحارة لأهالي المتوفيين، متمنيين الشفاء العاجل للمصابين، اللهم خفف عليهم مصابهم وثبت لهم الآجر."],
  ["https://twitter.com/Mofa_Libya", "1555688484396040193", "ندعوا المجتمع الدولي بالتحرك العاجل والفاعل لوقف التصعيد وتحمل مسؤوليته القانونية والأخلاقية إزاء الشعب الفلسطيني وتوفير الحماية له ، تجدد دولة #ليبيا موقفها الثابت من القضية الفلسطينية والحقوق المشروعة للشعب الفلسطيني الشقيق."],
  ["https://twitter.com/Mofa_Libya", "1555688334533558272", "تعرب وزارة الخارجية والتعاون الدولي بحكومة الوحدة الوطنية عن إدانتها واستنكارها الشديدين للعدوان الإسرائيلي على قطاع غزة مما أسفر عن سقوط شهداء وجرحي بينهم نساء وأطفال. https://t.co/Ijg2BG6F1p"],
......],

"evidence": [["https://twitter.com/Mofa_Libya", "1554448815524139013", "RT @NajlaElmangoush: أتقدم بالشكر لفخامة رئيس جمهورية #تونس السيد قيس سعيد @TnPresidencyعلى تضامنه وتسخير كل المستشفيات والأطقم الطبية لمساعدة جرحى #بنت_بيه وهذا التضامن يدل على أن ما يجمع الشعبين الشقيقين هو علاقات أخوية وروح تضامنية في السراء والضراء في كل الحالات @OJerandi\n 🇱🇾🇹🇳"],
  ["https://twitter.com/Mofa_Libya", "1554446617356427266", "1/2 وزارة الخارجية والتعاون الدولي تعرب عن شكرها وامتنانها العميق لما أعلنت عنه دولة #تونس الشقيقة في بيانها الأخير الذي سخرت فيه مستشفياتها وأطقمها الطبية التونسية؛ لمساعدة الليبيين الذين أصيبوا في بلدية #بنت_بيه إثر إنفجار صهريج الوقود. https://t.co/oWRtH9T7IC"]
....]
},
...
{
  "id": "AuRED_078",
  "rumor": "منظمة الصحة العالمية تدعو لوقف منح الجرعة الثانية من لقاحات كورونا حتى سبتمبر المقبل ما يسمح بايصال الجرعة الاولى من اللقاح للفئات الاكثر " ,
  "label": "NOT ENOUGH INFO",
  "timeline": [["https://twitter.com/DrTedros", "1421857856522002437", "RT @BahraintvNews: الجهود الوطنية للتصدي لفيروس كورونا في مملكة البحرين تبهر المدير العام لمنظمة الصحة العالمية خلال زيارته للمملكة .@WHO @DrTedros  @BDF_Hospital#وزارة_الإعلام  #bahrain  #كورونا_في_البحرين #كورونا  #البحرين #المنامة #صوت_الوطن_وعين_الحدث"],
  ["https://twitter.com/WHOEMRO", "1424064461611147274", "قم بزيارة صفحتنا الجديدة "شركاء في الصحة" عن المملكة العربية #السعودية، الشريك الاستراتيجي القديم ل @WHO وأحد أكبر المانحين، ذات السجل الحافل في دعم المبادرات الصحية العالمية المنقذة للحياة وعمليات الطوارئ.معًا من أجل تحقيق #الصحة_للجميع_وبالجميع https://t.co/0WAwx9mtF1"],
  ["https://twitter.com/WHOEMRO", "1423416531392806920", "❌الادعاء:  ينبغي على كل من تلقى لقاح كوفيد-19 الامتناع عن أخذ أي نوع من أنواع التخدير.✅الحقيقة: في الوقت الحالي، لا توجد أدلة علمية تؤيد أن التخدير يهدد الحياة أو غير آمن للاستخدام بعد تلقي لقاح كوفيد-19.لمزيد من حقائق اللقاح:➡️https://t.co/K7QtTVvBOK https://t.co/eFnCoVF9Jq"],
  ["https://twitter.com/WHOEMRO", "1423261810082426886", "ليس من الأسلم أن تُعطي رضيعك بدائل لبن الأم إذا كنتِ مصابة بمرض #كوفيد_19 إصابةً مؤكدة أو مُشتبهًا فيها 🤱https://t.co/wgp0yMCGnM\n#الأسبوع_العالمي_للرضاعة_الطبيعية https://t.co/B58EIK215r"]
...],

"evidence": []
},

...

```

**Examples for English data**:

```
{
  "id": "AuRED_089",
  "rumor": "The Corona epidemic has reached the Emirates, with 75 cases in Abu Dhabi and 63 cases in Dubai. A warning to refrain from traveling to the Emirates in order to preserve safety and not transmit the epidemic. May Allah protect Muslims everywhere..." ,
  "label": "REFUTES"
  "timeline": [["https://twitter.com/WHOEMRO", "1222971333522468867", "The World Health Organization declares the outbreak of the new #Coronavirus a public health emergency of international concern https://t.co/pVOXpZaPH7"],
   ["https://twitter.com/WHOEMRO", "1223608938136047616", "s. Do pneumonia vaccines protect against the new #Coronavirus? C. no. Pneumonia vaccines do not protect against the new coronavirus. This virus is new and different and needs its own vaccine. Researchers are working to develop a vaccine against this virus. #Know_the_facts https://t.co/QTGmI2flo9"],
   ["https://twitter.com/mohapuae", "1223361274618183681", "Learn about the symptoms of the new Corona virus #NewCoronavirus #Coronavirus #Corona #Ministry of Health and Community Protection https://t.co/jWALFtA68m"],
   ["https://twitter.com/mohapuae", "1223279618372882432", "Excerpts from the participation of the Ministry of Health and Community Protection in the Arab Health Exhibition and Conference 2020 through a group of innovative healthcare initiatives and projects under the slogan “Emirates Health is a Shared Responsibility” #Ministry_of_Health_and_Community_Protection_Emirates #Arab_Health_Exhibition_and_Conference_2020 #Emirates_Health https://t.co/c69pHj6ffd"],
   ......],

"evidence": [["https://twitter.com/WHOEMRO","1222506828694794240","Today @WHO confirmed the emergence of the first cases of the new Coronavirus in the Eastern Mediterranean Region, in the United Arab Emirates. Following @mohapuae's confirmation on January 29. 4 members of the same family from the Chinese city of Wuhan arrived in the UAE at the beginning of January 2020, and were admitted to the hospital after they were confirmed infected with the #Corona_virus."],
   ["https://twitter.com/mohapuae", "1222476311291142145", "Four people from a Chinese family were infected with the new Corona virus, all of whom are in stable condition and were contained according to internationally approved precautionary measures."]
  ,....]
},
...,
{
  "id": "AuRED_105",
  "rumor": "Tunisia offers to help Libya treat those injured in the fuel truck explosion in Bint Biyya #Libya #Witness for details: https://t.co/s7fdU5fvgq" ,
  "label": "SUPPORTS",
  "timeline": [["https://twitter.com/NajlaElmangoush", "1554448728320344064", "I extend my thanks to His Excellency the President of the Republic of #Tunisia, Mr. Kais Saied @TnPresidency, for his solidarity and harnessing all hospitals and medical teams to help the wounded of #BintBey. This solidarity indicates that what unites the two brotherly peoples is fraternal relations and a spirit of solidarity in good times and bad in all cases @OJerandi 🇱🇾🇹🇳"],
  ["https://twitter.com/NajlaElmangoush", "1554027191788355584", "The municipality of #BintBiya woke up at dawn today to a humanitarian catastrophe and terrible news, as a result of a fuel tanker explosion, which resulted in the death of 5 people and the injury of nearly 50 others. I offer our deepest condolences to the families of the deceased, wishing a speedy recovery to the injured. May Allah ease their affliction and grant them reward."],
  ["https://twitter.com/Mofa_Libya", "1555688484396040193", "We call on the international community to take urgent and effective action to stop the escalation and assume its legal and moral responsibility towards the Palestinian people and provide them with protection. The State of #Libya renews its firm position on the Palestinian issue and the legitimate rights of the brotherly Palestinian people."],
  ["https://twitter.com/Mofa_Libya", "1555688334533558272", "The Ministry of Foreign Affairs and International Cooperation of the National Unity Government expresses its strong condemnation and denunciation of the Israeli aggression on the Gaza Strip, which resulted in martyrs and wounded, including women and children. https://t.co/Ijg2BG6F1p"],
......],

"evidence": [["https://twitter.com/Mofa_Libya", "1554448815524139013", "RT @NajlaElmangoush: I extend my thanks to His Excellency the President of the Republic of #Tunisia, Mr. Kais Saied @TnPresidency, for his solidarity and harnessing all hospitals and medical teams to help the wounded of #Bint_Bey. This solidarity indicates that what unites the two brotherly peoples is fraternal relations and a spirit of solidarity in good times and bad in all cases @OJerandi 🇱 🇾🇹🇳"],
  ["https://twitter.com/Mofa_Libya", "1554446617356427266", "1/2 The Ministry of Foreign Affairs and International Cooperation expresses its deep thanks and gratitude for what the sisterly state of #Tunisia announced in its recent statement, in which it made use of its Tunisian hospitals and medical staff. To help Libyans who were injured in the municipality of #Bint_Bey following the fuel tanker explosion. https://t.co/oWRtH9T7IC"]....]
},
...
{
  "id": "AuRED_078",
  "rumor": "The World Health Organization calls for stopping the granting of the second dose of Corona vaccines until next September, which will allow the first dose of the vaccine to be delivered to the groups that need it most in the world." ,
  "label": "NOT ENOUGH INFO",
  "timeline": [["https://twitter.com/DrTedros", "1421857856522002437", "RT @BahraintvNews: The national efforts to combat the Corona virus in the Kingdom of Bahrain impress the Director-General of the World Health Organization during his visit to the Kingdom. . . @WHO @DrTedros @BDF_Hospital . . #Ministry_of_Information #bahrain #Corona_in_Bahrain #Corona #Bahrain #Manama #Voice_of_the_Nation_and_Ain_Hadath"],
  ["https://twitter.com/WHOEMRO", "1424064461611147274", "Visit our new “Partners in Health” page about #SaudiArabia, a long-standing strategic partner of @WHO and one of our largest donors, with a proven track record of supporting life-saving global health initiatives and emergency operations. Together for #HealthForAll, By All https:/ /t.co/0WAwx9mtF1\""],
  ["https://twitter.com/WHOEMRO", "1423416531392806920", "❌Claim: Anyone who has received the Covid-19 vaccine should refrain from taking any type of anesthesia. ✅Fact: Currently, there is no scientific evidence to support that anesthesia is life-threatening or unsafe to use after receiving the COVID-19 vaccine. For more vaccine facts: ➡️https://t.co/K7QtTVvBOK https://t.co/eFnCoVF9Jq"],
  ["https://twitter.com/WHOEMRO", "1423261810082426886", "It is not safe to give your baby breastmilk substitutes if you have a confirmed or suspected case of #Covid_19 🤱https://t.co/wgp0yMCGnM #WorldBreastfeedingWeek https://t.co/B58EIK215r"]
....],

"evidence": []
},

...

```



   
