import asyncio
import bz2
import itertools
import logging
from typing import Iterator, Sequence
import os

from bs4 import SoupStrainer  # type: ignore[attr-defined]
from lxml import etree

from langchain_community.document_loaders import (
    BSHTMLLoader,
    PyPDFLoader,
    WebBaseLoader,
    JSONLoader,
)
from langchain_core.documents import Document


def json_loader(file_path: str) -> list[Document]:
    loader = JSONLoader(
        file_path,
        jq_schema=".[]",
        content_key="content",
        metadata_func=lambda record, meta: record.get("metadata", {}),
    )
    return loader.load()


def html_loader(file_path: str) -> list[Document]:
    loader = BSHTMLLoader(file_path)
    return loader.load()


def pdf_loader(file_path: str) -> list[Document]:
    loader = PyPDFLoader(file_path)
    docs = asyncio.run(loader.aload())
    return docs


def sync_pdf_loader(file_path: str) -> list[Document]:
    all_documents = []
    for file in os.listdir(file_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(
                os.path.join(file_path, file), mode="single"
            )  # important: default of mode is page-vise!
            docs = loader.load()
            all_documents.extend(docs)
    return all_documents


def web_loader(web_path: Sequence[str]) -> list[Document]:
    bs4_strainer = SoupStrainer(
        class_=[
            "SP-Content__main",
            "section sectionZ sectionArticle",
            "page__content",
            "post-content",
        ]
    )
    loader = WebBaseLoader(
        web_path=web_path,
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    return docs


def cleanup_etree(elem: etree._Element) -> None:
    elem.clear()
    while elem.getprevious() is not None:
        del elem.getparent()[0]  # type:ignore[union-attr]


def read_wikibooks_dump(path: str) -> Iterator[tuple[str, str]]:
    total = 0
    skipped = 0
    file = bz2.open(path)
    for action, elem in etree.iterparse(
        file, events=("end",), recover=True, huge_tree=True
    ):
        localname = etree.QName(elem).localname
        if localname == "page":
            title = elem.xpath('*[local-name()="title"]')
            text = elem.xpath('*//*[local-name()="text"]')
            if len(title) == 1 and len(text) == 1 and text[0].text is not None:
                yield title[0].text, text[0].text
                total += 1
                if total % 10000 == 0:
                    logging.debug("Pages read from %s: %d", path, total)
            else:
                skipped += 1
            cleanup_etree(elem)
    logging.debug("Pages skipped in %s: %d", path, skipped)


def wikibooks_loader(path: str, limit: int | None = None) -> list[Document]:
    docs: list[Document] = []
    for title, text in itertools.islice(read_wikibooks_dump(path), limit):
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": f"wikibooks:{title}",
                },
            )
        )
    return docs
