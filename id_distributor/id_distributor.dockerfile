FROM node:22-alpine AS id_distributor
WORKDIR /app
COPY src/id_distributor.js src/
COPY package-lock.json .
COPY package.json .
RUN npm i
EXPOSE 8621
CMD ["npm", "run", "start"]