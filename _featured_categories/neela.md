---
layout: about
description: >
  About Matt Taylor.
hide_description: true
redirect_from:
  - /download/
---

# About

<!--author-->

## Work Experience 

{% assign work = site.work_experience | sort: "year" | reverse %}
{% for job in work %}
<div class="pubitem">
    <div class="pubteaser">
        <a href="{{job.url}}">
            <img
            src="/assets/img/work/{{ job.slug }}.png"
            alt="{{job.slug}} logo"
            />
        </a>
    </div>
  <div class="pubtitle">{{ job.title }}</div>
  <div class="pubauthors">{{ job.job_date }}</div>
  <br>
  <div class="pubinfo">Mentor: <a href="{{job.mentor_webpage}}" target="_blank" style="text-decoration: underline;">{{ job.mentor }}</a></div>
</div>
{% endfor %}



