import argparse
import logging
import requests

from sombrero import Detector


def main():
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default='resources/params.yml')
    parser.add_argument("--url", required=True, type=str, default='')
    args = parser.parse_args()

    # Webpage to extract content from.
    resp = requests.get(args.url)
    if resp.status_code >= 400:
        logging.error("Could not fetch {}".format(args.url))
        return

    detector = Detector(args.params)
    sentences = detector.run(resp.text)

    for x in sentences:
        print(x)


if __name__ == '__main__':
    main()
