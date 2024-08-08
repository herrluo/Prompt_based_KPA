import os

import pandas as pd
from techniques import Technique
from keybert import KeyBERT
from ablation_study.clustering.clustering import get_clusters_contents, get_cluster_members
import hashlib
import ast

kw_model = KeyBERT()

lock = False
prefix = 'demo'

metrics = [
    'rouge',
    'bart_score',
    'bert_score',
    'bleu',
    'meteor',
    'chrf',
    'blanc',
    'rouge_we',
    'mover_score',
    'summa_q',
    'coverage',
    'density',
    'compression',
    'summary_length',
    'percentage_novel',
    'percentage_repeated',
    'perplexity',
    'cider',
    'success_rate',
    'sentence_movers',
    'supert'
]

max_kps_demo = 10


def get_kps_for_text(input_text, technique_name, hashed_text):
    global prefix
    cluster_res_path = os.path.join(os.path.abspath('..'), 'clustering_results',
                                    f'{prefix}_{technique_name}_{hashed_text}.csv')
    # extract clusters cluster_top_text, cluster_full_text
    layer_depth = 4
    layer_size = 4
    top_n = 2
    n_tries = 10
    top_keywords = 10
    kp_min_size = 3
    layer_min_size = 0.1

    ct_df, ct = get_clusters_contents(
        input_text,
        layer_depth=layer_depth,
        layer_size=layer_size,
        top_n=top_n,
        kp_min_size=kp_min_size,
        layer_min_size=layer_min_size
    )

    key_points_df = pd.DataFrame()
    key_points = []
    technique = Technique(technique_name, n_tries)
    # get only top 10 clusters or even less
    kp_df = ct_df[ct_df['tag_name'] == 'Key Point']
    kp_df.sort_values(by=['value'], ascending=False, inplace=True)
    kp_df = kp_df.head(max_kps_demo)
    for i, row in kp_df.iterrows():
        prompt = row['labels'] + ' TL;DR '
        cluster_id = row['id']
        members = row['cluster_members']
        cluster_text, _ = ct.cluster_summary_simple(members, len(members))
        cluster_text = [i.tolist() for i in cluster_text]
        # remove duplicates
        cluster_text = list(set(cluster_text))
        cluster_text = ' '.join(cluster_text)
        cluster_text = ' '.join(cluster_text.split()[:700])
        extracted_keywords = kw_model.extract_keywords(cluster_text, top_n=top_keywords, use_mmr=True)
        extracted_keywords = [i[0] for i in extracted_keywords]
        best_sentence = technique.generate_summary(prompt, extracted_keywords, cluster_text, None)
        if best_sentence is None:
            continue
        generated_sum = best_sentence['generated_sum'].iloc[0]
        key_points_df = pd.concat([key_points_df, best_sentence])
        key_points.append(generated_sum)
        ct_df.loc[ct_df['id'] == cluster_id, 'labels'] = generated_sum
        ct_df.loc[ct_df['id'] == cluster_id, 'status'] = 'processed'
        for k in metrics:
            if k in best_sentence.columns:
                ct_df.loc[ct_df['id'] == cluster_id, k] = best_sentence[k].iloc[0]
        ct_df.loc[ct_df['id'] == cluster_id, 'cluster_text'] = cluster_text
        ct_df.loc[ct_df['id'] == cluster_id, 'prompt'] = prompt
        ct_df.loc[ct_df['id'] == cluster_id, 'extracted_keywords'] = str(extracted_keywords)
        ct_df.to_csv(cluster_res_path, index=False)

    ct_df.to_csv(cluster_res_path, index=False)
    values = {}
    for k in metrics:
        if k in key_points_df.columns:
            values[k] = key_points_df[k].mean()
    values['generated_sum'] = '. '.join(key_points)
    values['technique'] = technique_name
    return {
        "results": pd.DataFrame(values, index=[0]),
        "path": cluster_res_path
    }


def live_inference(full_text, technique_name):
    global lock
    global prefix
    if lock:
        return None
    lock = True
    hashed_text = int(hashlib.sha1(full_text.encode("utf-8")).hexdigest(), 16) % (10 ** 8)
    # df_name = fr'D:\backup_user\crypto\thesis\my-repos\cryptocurrencies-kpa\Final\results\{prefix}_{technique_name}_results.csv'
    df_name = os.path.join(os.path.abspath('..'), 'results', f'{prefix}_{technique_name}_results.csv')
    all_df = pd.DataFrame()
    kps = get_kps_for_text(full_text, technique_name, hashed_text)
    best_sentence = kps["results"]
    best_sentence['text'] = full_text
    all_df = pd.concat([all_df, best_sentence])
    all_df.to_csv(df_name, index=False)
    lock = False
    return kps["path"]


if __name__ == '__main__':
    test_text = "the right to die is a personal choice and everyone has the right to make it happen if they truly wish. assisted suicide is a blessing not a crime  a person who wants to be assisted in this is in great pain or has no ability to be cured  let someone give the suffering the chance to end it . when people are in unbearable pain and suffering and have no chance of getting better it is cruel to keep them alive without their consent. many people are suffering terribly but are physically incapable of dying on their own. Assisted suicide should not be a criminal offence as people should not be expected to live in vegetative states and need help to make their own decisions to die a reality . Sometimes people are so ill they have no quality of life and want to end it themselves but are physically unable ot do so  in thee cases it is only merciful for it to be legal for someone to help them . assisted suicide should not be a criminal offense because when a person get gravely ill they should not have to keep on suffering . assisted suicide helps terminally ill people end their suffering. assisted suicide should not be a criminal offence  because if a loved one is terminally ill  it provides them with the opportunity to die with dignity rather than suffering until they die naturally . if someone is suffering so much that they see suicide as the only way out  then criminalising their behaviour is likely to do more harm than good . if a terminally ill patient does not have assisted suicide available they may turn to another more violent method of ending their own life . We don t allow our pets to suffer but because of religious doctrine we force humans to die in mental anguish and pain instead of decriminalising assisted suicide so they can die with dignity . If a person is already diagnosed with only a short time to live and is undergoing excruciating pain  assisted death is a compassionate support for the person . people should be allowed to die with dignity instead of suffering a long painful death . there is no reason for dying people to suffer unnecessarily. assisted suicide is a humane and caring way to allow a person to help another person whose life is too difficult to be lived properly in any manner  so it should not be criminalized . Assisted suicide helps those who are in pain due to a devastating disease end their own lives on their own terms . assisted suicide should not be a criminal offense because when you get so sick with no hope left  one should be able to end the pain . assisted suicide should not be a criminal offence because it helps to ease the suffering of those who will never recover from their illnesses or injuries . being able to die with dignity should be a right  when someone thinks their time has come  they should be able to die comfortable without worrying if having their loved ones nearby would get them jail time . people that are terminally ill should be able to choose whether they want to have to live suffering or whether they want to end the pain . people who have terminal illness or suffer from certain medical conditions should have the freedom to choose and if medical personnel helps them it shouldn t be treated as offence . it is more humane to help the person die than to let them suffer excruciating pain. Assisted suicide and the right to live with dignity is a human right and should be made legal everywhere  It is a human right to choose not to suffer with a terminal illness . Assisted suicide should not be a criminal offence as people with no quality of life should not have to suffer indefinitely and their trusted helper should not be a criminal for helping them . assisted suicide is a compassionate way to help people choosing to end their life in a nonviolent way . a person should not be forced into having to live their lives suffering  when death is already enevitable. assisted suicide shouldn t be a criminal offence because every human being should have the right to have someone assist them in passing should they determine they no longer wish to live. people have the right to die on their own terms and in their own time. assisted suicide would save terminally i ll people pain and suffering . the end of life can be painful and humiliating  and we should allow people the right to decide when they are ready to end their own suffering . If a person was terminally ill and no treatments were working on them and they have only weeks to live  than assisted suicide should be a way out . people who are terminally ill with no hope of survival should be spared the pain and suffering of living . assisted suicide shouldn t be a criminal offence if they are suffering from an incurable disease  they are going to die anyway. to enable a very sick individual to end their life should be seen as an act of kindness not a criminal offence as it takes great strength of character to be that enabler in what is a very difficult time . assisted suicide would relieve much suffering by those with a terminal illness that is too much to bear . a person has the right to end their suffering and if somebody takes pity on them and chooses to help  that person should not be punished . assisted suicide should not be a criminal offence because if someone doesn t want to live they shouldn t be forced to. Death with dignity is having the option to decide your own fate  You re able to take control of your own fate  It allows you to leave the world with your true self intact . assisted suicides are less painful and are an act of mercy. assisted suicide should not be a criminal offense because if someone is in enough pain and is sane enough to decide  they should have that choice . assisted suicide allows for a person who is sick and fully understanding what is done to end their pain when no quality of life can be assured. assisted suicide gives people control over their life in terrible circumstances  it allows them dignity in death rather than living in terrible conditions for many years . assisted suicide should not be a criminal offense because it helps people who are in constant pain due to a terminal illness. if someone wants to die they shouldn t have to stay alive  so helping them to die is the beat thing to do  that shouldn t be a criminal offence . people should have the right to die  some people who are severely suffering without hope should be able to end their lives  sometimes this cannot be done without assistance . people must be able to decide what to do or not with their lives therefore assisted suicide is an alternative. a person should have the right to die on their own terms. assisted suicide is a lot better than letting a person suffer until end of life . people who are terminally ill and don t want to suffer any more have a right to ask for help to end their lives. People should be able to ask for help in ending their own life if they are living with an incurable disease . assisted suicide should not be a criminal offence if the patients asking for it to be done to them. assisted suicide should not be a criminal offense because if a person doesn t want to live  why should they be forced to. assisted suicide should not be a crime  if someone is suffering on a daily basis from something that they are going to die from anyway  they should be allowed to die on their own terms . assisted suicide usually involves a terminal medical patient that is tired of fighting and being in agony  they should be allowed to  check out  and receive help doing so . some patients have the utmost suffering and must end it as soon as possible.  people reach their limit when it comes to their quality of life and should be able to end their suffering  this can be done with little or no suffering by assistance and the person is able to say good bye . assisted suicide shouldn t be a criminal offence because it allows the patient to have control over their final decisions. assisted suicide allows people who have terrible health conditions to die with dignity  if assisted suicide is a crime they may have to endure many more years living in agony . people have a basic right to bodily autonomy  deciding whether or not to die with minimal suffering and dignity is integral to that right . if a patient is suffering with cancer or other painful conditions  they should have a right to end their suffering. when people are suffering  it is unfair to force them to continue to do so  if someone can assist them in ending the suffering  they should be able to . if a person who is dying anyway wants to end their life in a peaceful way then they should be able to do so . assisted suicide sometimes is the only thing that ends suffering . if somebody is going to commit suicide anyway they should be able to do it safely with assistance. assisted suicide is a great help for people who suffer from deadly diseases and are on their deathbed. people should be free to help people out of their pain by helping them commit suicide without the fear of prosecution. assisted suicide should be made legal many people suffer with debilitating and painful illnesses and should be counselled and supported to make an informed choice. if someone has an illness and is in constant pain  helping them end their misery is the humane thing to do and should be understood . people who are terminally ill and suffering greatly should have the right to end their own life if they so desire . people with terminal illnesses are going to die  by not allowing assisted suicide we are making these people suffer deeply which in inhumane . assisted suicide must be allowed  if a terminally ill person wants to end their life  allow them the peace of being able to with the help of a professional instead of potentially making it worse on their own . people have the right to their bodies and if they wish to die they should be able to get help to do so . people willing to die  it s their choice  should be able to leave this world in a civil way and helped with assisted suicide when needed. A patient should be able to decide when they have had enough  care  . assisted suicide should be made legal so that those who have a terminal illness or are in uncurable pain can make an informed choice to end their lives with dignity surrounded by those they love . If a person only has weeks to live  then assisted suicide may be the answer if all other possibilities have been exhaussted . it helps relieve some people from tremendous suffering. it is unfair to force someone with a terminal illness to continue to suffer greatly when they are just dying anyway  the only humane and compassionate way to help them is to allow assisted suicide . to assist a loved one to pass on and be relieved of suffering is a selfless act that should not be criminalised . people who are suffering with terminal illnesses with no hope of cure and in immense pain should be allowed to end their suffering  it s safer and painless done in a doctor s care rather than their own devices . assisted suicide shouldn t be a criminal offence because sometimes the person is in such pain or has no life just laying around in a bed. terminally ill patients who just don t want to continue suffering have a valid reason to contemplate suicide  and someone who wants to help ease their pain should not be punished . many people face a cruel and protracted period of suffering before their death so with compassion they should be helped therefore assisted suicide should not be a criminal offence . if to a reasonable degree of medical certainty the medical condition of a person is irreversible  it must have the right to terminate his life by requesting assistance without consequences for the fact . assisted suicide is a personal choice on how a person wants to live his or her life and therefore should be legal by the constitution. assisted suicide is a humane way to end a life as we do with animals in pain and should not be a criminal offence . assisted suicide should be allowed because it allows people with terminal illnesses to die with dignity before suffering a long and painful death . assisted suicide allows terminally ill persons to choose when to die and do it with dignity  it lets people decide not to live in a state of pain and misery and choose to pass on with human intervention . assisted suicide allows one to end terminal pain and suffering and should be allowed to continue . assisted suicide should not be a criminal offence because it is sometimes the only way a person suffering a debilitating medical condition such as motor neurone disease can end there life . assisted suicide should not be a crime since the perpetrator was merely obeying the wishes of another . sometimes people who are unable to end their own lives  are in so much pain and anguish that it is cruel to make them keep living  their loved ones need to be able to help them legally . if someone is suffering from an incurable disease and is in great pain  it can be humane to help end their suffering . if people has a life ending illness they should be allowed to die. if to a degree of reasonable certainty the state of seriousness can not be reversed by the medicine  the patient must have the right to end his life and if he needs help for that he should not be condemned. people should have the right to die if they choose . people should have the right to decide how and when they want to die  particularly if they are suffering from a terminal illness. People and family members must decide what they want to do with their life and body. people should be allowed to die with dignity if they choose to . people who have no hope of living should have a choice to end their life. assisting a suicide for medical reasons should not be considered a criminal offense because the person is making sound judgements based on someone else s suffering . it shouldn t be an offence because is someone is seriously ill and in pain they have the right to end their own life if they are suffering . people whose quality of life is so poor that death would be preferable should be allowed to choose assisted suicide to end their suffering and pass with dignity. people should choose what to do with their lives  even if it means ending it in whatever manner they choose. for those with degenerative diseases  assisted suicide is often the only solace they have. assisted suicide should no be considered a criminal offence because a person could really be suffering and in pain and to let them suffer is heartless and inhumane. a person should have the dignity to choose how they die. there are people who suffer from terrible illnesses who are mentally capable of saying  enough is enough   there is a realization that quality of life is so poor that death is a better option . everyone should have the right to chose the direction they want to take  if someone is in unbearable pain with no quality of life then they should have the right to seek assistance . assited suicide allows those with a painful and terminal illness the ability to chose to end their life to cease the pain and suffering that would last for the end of their lives . assisted suicide is sometimes the only solution for some people  we should not take it off . assisted suicide should not be a criminal offense since it should be a personal decision what a person wants for themselves  it could prevent months or years of pain for someone with no quality of life . assisted suicide allows terminally ill people to die with dignity and should not be criminalized. a person should have the right to be able to choose if they want to live or die. Some diseases are life threatening and can put a person in a no living state  These situations are hard to bear for the ill person and for his family and relatives  only assisted suicide can make this ease. assisted suicide is a human right and would spare the terminally i ll pain . assisted suicide should not be a criminal defense because in some cases  like people who have terminal cancer  you are doing them a mercy and saving them from suffering . assisted suicide is necessary in our modern world full of people dealing with pain and sorrow. assisted suicide should be allowed in certain circumstances such as when a person is suffering from severe illnesses . assisted suicide should not be a criminal offence because a person should be able to decide how and when they want t die "
    live_inference(test_text, 'flan')
